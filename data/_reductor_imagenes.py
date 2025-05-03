# -*- coding: utf-8 -*-

import os
import sys
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageFilter
# from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QMessageBox  # Eliminado
# from PyQt5.QtGui import QIcon  # Eliminado
# import winreg  # Eliminado
# import ctypes  # Eliminado
import filecmp
import difflib
from concurrent.futures import ThreadPoolExecutor
import gc

# Variable para habilitar o deshabilitar el uso de batch
USE_BATCH = False

print("Iniciando reductor de imágenes...")


# Corregir la advertencia de depreciación usando la constante actualizada
try:
    from PIL import Image
    RESAMPLING_METHOD = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.Resampling.LANCZOS
except:
    # Fallback para versiones muy antiguas
    RESAMPLING_METHOD = Image.ANTIALIAS

# Para aceleración por hardware NVIDIA (si está disponible)
try:
    import torch
    import torchvision.transforms as transforms
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

class ImageReducer:
    def __init__(self, source_folder):
        self.source_folder = Path(source_folder)
        self.folder_name = self.source_folder.name
        self.dest_folder = None
        self.needs_processing = False
        self.has_reduced_images = False
        self.create_dest_folder()
        
        # Configurar logging
        self.log_file = self.dest_folder / "image_reduction_log.txt"
        os.makedirs(self.dest_folder, exist_ok=True)
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')
        
        # Registro de procesamiento
        self.processed_registry = self.source_folder / "processed_folders_registry.txt"
        
        # Contadores
        self.total_processed = 0
        self.total_reduced = 0
        self.total_copied = 0
        self.errors = 0
        
        # Para verificación de carpetas
        self.processed_folders = []
        self.different_files = []

    def create_dest_folder(self):
        """Crea el nombre correcto para la carpeta de destino según las reglas"""
        folder_name = self.folder_name
        
        # Verificar si la carpeta necesita procesamiento
        self.needs_processing = self.folder_contains_large_images(self.source_folder)
        
        # Lógica mejorada para nombrar carpetas
        if not self.needs_processing:
            if folder_name.endswith("- g -"):
                # No hay imágenes para reducir y ya tiene "- g -"
                dest_folder_name = f"{folder_name.rsplit('- g -', 1)[0].strip()} -- "
            elif folder_name.endswith("-"):
                # No hay imágenes y termina en guión
                dest_folder_name = f"{folder_name.rstrip('-').strip()} -- "
            else:
                # No hay imágenes y no tiene formato especial
                dest_folder_name = f"{folder_name} -- "
        else:
            # Hay imágenes que necesitan reducción
            self.has_reduced_images = True
            
            if folder_name.endswith("- g -"):
                # Ya tiene "- g -", añadimos otro
                dest_folder_name = f"{folder_name} g -"
            elif folder_name.endswith("-"):
                # Termina en guión
                dest_folder_name = f"{folder_name.rstrip('-').strip()} - g -"
            else:
                # Caso normal
                dest_folder_name = f"{folder_name} - g -"

        
        self.dest_folder = self.source_folder.parent / dest_folder_name

    def folder_contains_large_images(self, folder_path):
        """Verifica si la carpeta o subcarpetas contienen imágenes grandes"""
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    file_path = Path(root) / file
                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                            if width > 1920 or height > 1920:
                                return True
                    except Exception as e:
                        # Ignorar errores al verificar
                        pass
        return False

    def reduce_image(self, image_path):
        """Redimensiona una imagen si es necesario, con soporte para GPU si está disponible"""
        try:
            # Liberar memoria explícitamente antes de procesar
            gc.collect()
            
            # Usar aceleración por hardware si está disponible
            if HAS_CUDA:
                return self.reduce_image_cuda(image_path)
            else:
                return self.reduce_image_cpu(image_path)
        except Exception as e:
            logging.error(f"Error procesando {image_path}: {e}")
            return False

    def reduce_image_cpu(self, image_path):
        """Redimensiona una imagen usando CPU"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Condición para redimensionar
                if width > 1920 or height > 1920:
                    # Manejar errores de memoria
                    try:
                        # Mantener proporción
                        img.thumbnail((1920, 1920), RESAMPLING_METHOD)
                        
                        # Guardar en nueva ubicación
                        relative_path = image_path.relative_to(self.source_folder)
                        new_path = self.dest_folder / relative_path
                        
                        # Crear directorios si no existen
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        img.save(new_path, quality=90)
                        logging.info(f"Redimensionada: {relative_path}")
                        return True
                    except MemoryError:
                        # Si hay error de memoria, intentar con modo progresivo
                        logging.warning(f"Memoria insuficiente para {image_path}, usando modo progresivo")
                        img = Image.open(image_path)
                        img.thumbnail((1920, 1920), Image.Resampling.NEAREST)
                        
                        relative_path = image_path.relative_to(self.source_folder)
                        new_path = self.dest_folder / relative_path
                        
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        if image_path.suffix.lower() == '.jpg' or image_path.suffix.lower() == '.jpeg':
                            img.save(new_path, quality=85, optimize=True, progressive=True)
                        else:
                            img.save(new_path)
                        
                        logging.info(f"Redimensionada (modo alternativo): {relative_path}")
                        return True
                return False
        except Exception as e:
            logging.error(f"Error CPU procesando {image_path}: {e}")
            return False

    def reduce_image_cuda(self, image_path):
        """Redimensiona una imagen usando GPU CUDA"""
        try:
            # Abrir la imagen con PIL primero para verificar dimensiones
            with Image.open(image_path) as pil_img:
                width, height = pil_img.size
                
                # Condición para redimensionar
                if width > 1920 or height > 1920:
                    # Procesamiento con GPU
                    img = Image.open(image_path).convert('RGB')
                    
                    # Convertir a tensor
                    tensor = transforms.ToTensor()(img).unsqueeze(0)
                    
                    # Mover a GPU
                    if torch.cuda.is_available():
                        tensor = tensor.cuda()
                    
                    # Calcular nueva proporción
                    scale = min(1920 / width, 1920 / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    # Redimensionar usando GPU
                    resized = transforms.functional.resize(tensor, (new_height, new_width))
                    
                    # Mover de vuelta a CPU y convertir a PIL
                    resized = resized.squeeze(0).cpu()
                    resized_img = transforms.ToPILImage()(resized)
                    
                    # Guardar en nueva ubicación
                    relative_path = image_path.relative_to(self.source_folder)
                    new_path = self.dest_folder / relative_path
                    
                    # Crear directorios si no existen
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Guardar la imagen
                    resized_img.save(new_path, quality=90)
                    logging.info(f"Redimensionada (CUDA): {relative_path}")
                    
                    # Limpiar memoria GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    return True
                return False
        except Exception as e:
            logging.error(f"Error CUDA procesando {image_path}: {e}")
            # Fallback a CPU si falla CUDA
            return self.reduce_image_cpu(image_path)

    def process_images_with_batch(image_paths, batch_size):
        """Procesa imágenes en lotes usando CUDA."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {device}")

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img_tensor = transforms.ToTensor()(img)
                    images.append(img_tensor)
                except Exception as e:
                    logging.error(f"Error al cargar la imagen {path}: {e}")

            if images:
                batch = torch.stack(images).to(device)
                print(f"Procesando un lote de tamaño: {batch.size()}")
                # Aquí puedes realizar operaciones en el lote

    def process_folder(self):
        """Procesa recursivamente la carpeta"""
        print(f"Analizando carpeta {self.source_folder}...")
        logging.info(f"Iniciando procesamiento de {self.source_folder}")

        if not self.needs_processing:
            message = f"No hay imágenes grandes para reducir en {self.source_folder}"
            print(message)
            logging.info(message)
            return

        try:
            start_time = time.time()  # Inicio del cálculo de tiempo

            image_paths = []
            for root, _, files in os.walk(self.source_folder):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")):
                        image_paths.append(os.path.join(root, file))

            if USE_BATCH:
                self.process_images_with_batch(image_paths, batch_size=16)
            else:
                for image_path in image_paths:
                    self.reduce_image(image_path)

            elapsed_time = time.time() - start_time  # Fin del cálculo de tiempo
            print(f"Tiempo total transcurrido: {elapsed_time:.2f} segundos")
            logging.info(f"Tiempo total transcurrido: {elapsed_time:.2f} segundos")

            self.compare_folders()
            self.update_registry()

        except Exception as e:
            logging.error(f"Error general: {e}")
            print(f"Error: {e}")

        finally:
            logging.info("Procesamiento finalizado")
            print("========================")
            print("Procesamiento finalizado")
            print("========================")

    def compare_folders(self):
        """Compara las carpetas origen y destino para verificar integridad"""
        try:
            origin_file_count = sum([len(files) for _, _, files in os.walk(self.source_folder)])
            dest_file_count = sum([len(files) for _, _, files in os.walk(self.dest_folder)])

            # Ajustar el conteo de archivos destino para ignorar el archivo de log
            if self.log_file.exists():
                dest_file_count -= 1

            comparison_info = f"Comparación de carpetas:\n"
            comparison_info += f"Archivos en origen: {origin_file_count}\n"
            comparison_info += f"Archivos en destino: {dest_file_count}\n"

            if origin_file_count != dest_file_count:
                comparison_info += "¡ADVERTENCIA! El número de archivos no coincide\n"
            else:
                comparison_info += "El número de archivos coincide correctamente\n"

            # Mostrar la información de comparación
            print(comparison_info)
            logging.info(comparison_info)

        except Exception as e:
            logging.error(f"Error al comparar carpetas: {e}")

    def update_registry(self):
        """Guarda las carpetas procesadas en un archivo de registro"""
        try:
            with open(self.processed_registry, "a", encoding="utf-8") as f:
                # Agregar las carpetas procesadas al archivo
                for folder in self.processed_folders:
                    f.write(folder + "\n")
                print(f"Registro actualizado en {self.processed_registry}")
                logging.info(f"Registro actualizado en {self.processed_registry}")
        except Exception as e:
            logging.error(f"Error al actualizar el registro: {e}")
    

if __name__ == "__main__":
    import time  # Importar para usar pausas
    print("Inicio del programa")
    
    if HAS_CUDA:
        print("CUDA está disponible. Se utilizará para la reducción de imágenes.")
    else:
        print("CUDA no está disponible. Se utilizará la CPU para la reducción de imágenes.")

    if len(sys.argv) > 1:
        source_folder = sys.argv[1]
        print(f"Carpeta fuente recibida: {source_folder}")

        if not os.path.exists(source_folder):
            print(f"Error: La carpeta {source_folder} no existe.")
            sys.exit(1)
        
        image_reducer = ImageReducer(source_folder)
        print("Instancia de ImageReducer creada")

        image_reducer.process_folder()
        print("Procesamiento de carpeta completado")
    else:
        print("Por favor, elige una o más carpetas y usa la opción de menú contextual \"Reducir imagenes\".")

