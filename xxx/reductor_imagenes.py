# -*- coding: utf-8 -*-

import os
import sys
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageFilter
import filecmp
import difflib
from concurrent.futures import ThreadPoolExecutor
import gc

# Variable para habilitar o deshabilitar el uso de multihilo
USE_MULTITHREAD = True

print("Iniciando reductor de imágenes...")

# Corregir la advertencia de depreciación usando la constante actualizada
try:
    from PIL import Image
    RESAMPLING_METHOD = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.Resampling.LANCZOS
except:
    # Fallback para versiones muy antiguas
    RESAMPLING_METHOD = Image.ANTIALIAS

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
        
        # Seguimiento de imágenes procesadas y no procesadas
        self.processed_images = set()
        self.failed_images = set()

    def generate_dest_folder_name(self, folder_name, needs_processing):
        """Genera el nombre de la carpeta destino según las reglas."""
        if not needs_processing:
            if folder_name.endswith("- g -"):
                return f"{folder_name.rsplit('- g -', 1)[0].strip()} -- "
            elif folder_name.endswith("-"):
                return f"{folder_name.rstrip('-').strip()} -- "
            else:
                return f"{folder_name} -- "
        else:
            if folder_name.endswith("- g -"):
                return f"{folder_name} g -"
            elif folder_name.endswith("-"):
                return f"{folder_name.rstrip('-').strip()} - g -"
            else:
                return f"{folder_name} - g -"

    def create_dest_folder(self):
        """Crea el nombre correcto para la carpeta de destino según las reglas"""
        folder_name = self.folder_name

        # Verificar si la carpeta necesita procesamiento

        print(f"Verificando si la carpeta tiene imagenes grandes...")
        self.needs_processing = self.folder_contains_large_images(self.source_folder)

        # solo muestra si hay imagenes grandes y detiene el proceso
        # print(f"Carpeta {self.source_folder} necesita procesamiento: {self.needs_processing}")
        # logging.info(f"Carpeta {self.source_folder} necesita procesamiento: {self.needs_processing}")
        # Termina el proceso aqui
        # exit(0)

        # Generar el nombre de la carpeta destino usando la función
        dest_folder_name = self.generate_dest_folder_name(folder_name, self.needs_processing)


        # CORRECCIÓN: Asegurarse de que la ruta no tenga barras dobles
        self.dest_folder = Path((self.source_folder.parent / dest_folder_name).as_posix())

        # eliminar espacios al final
        self.dest_folder = self.dest_folder.with_name(self.dest_folder.name.rstrip())

        # self.dest_folder = self.source_folder.parent / dest_folder_name

        # Aplicar las mismas reglas a las subcarpetas
        for root, dirs, _ in os.walk(self.source_folder):
            for dir_name in dirs:
                subfolder_path = Path(root) / dir_name
                subfolder_name = subfolder_path.name
                needs_processing = self.folder_contains_large_images(subfolder_path)

                # Generar el nuevo nombre de la subcarpeta
                new_subfolder_name = self.generate_dest_folder_name(subfolder_name, needs_processing)
                new_subfolder_path = subfolder_path.parent / new_subfolder_name
                # No renombrar las carpetas, solo registrar el nombre para uso futuro
                # subfolder_path.rename(new_subfolder_path)

    def _get_large_images(self, folder_path, max_size=1920, first_only=False, print_found=True):
        """Devuelve una lista de imágenes grandes (o la primera si first_only=True), sin multihilo. Muestra en consola cada archivo comprobado. Si print_found=False, no imprime el mensaje [ENCONTRADA]."""
        image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")):
                    image_files.append(Path(root) / file)
        if not image_files:
            return [] if not first_only else (None, None)
        resultados = []
        errores = []
        def check_image(image_path):
            # print(f"[CHECK] Comprobando tamaño de: {image_path}")
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width > max_size or height > max_size:
                        if print_found:
                            print(f"[ENCONTRADA] Imagen grande: {image_path} ({width}x{height})")
                        return (image_path, (width, height))
            except Exception as e:
                errores.append((image_path, str(e)))
            return None
        for path in image_files:
            result = check_image(path)
            if result:
                if first_only:
                    return result
                resultados.append(result)
        if errores:
            print(f"[ADVERTENCIA] No se pudieron abrir {len(errores)} imágenes:")
            for path, err in errores:
                print(f"  - {path}: {err}")
        if first_only:
            # Si no se encontró ninguna imagen grande, devolver (None, None)
            return (None, None)
        return resultados

    def folder_contains_large_images(self, folder_path):
        """Devuelve True si hay al menos una imagen grande en la carpeta o subcarpetas. No imprime mensaje [ENCONTRADA]."""
        result = self._get_large_images(folder_path, first_only=True, print_found=False)
        return result[0] is not None

    def find_large_image(self, folder_path, max_size=1920):
        """Devuelve la primera imagen grande encontrada y su tamaño, o (None, None) si no hay. Imprime mensaje [ENCONTRADA]."""
        return self._get_large_images(folder_path, max_size=max_size, first_only=True, print_found=True)

    def reduce_image(self, image_path):
        """Redimensiona una imagen si es necesario y muestra en consola cuál está procesando"""
        try:
            # print(f"[REDUCIENDO] Procesando imagen: {image_path}")
            gc.collect()
            # Si la imagen ya fue procesada exitosamente, saltar
            if str(image_path) in self.processed_images:
                return True

            result = self.reduce_image_cpu(image_path)
                
            # Registrar el resultado
            if result:
                self.processed_images.add(str(image_path))
                self.total_reduced += 1
            else:
                # Si no se necesita reducir, intentar copiarla
                copied = self.copy_image(image_path)
                if copied:
                    self.processed_images.add(str(image_path))
                    self.total_copied += 1
                    return True
                else:
                    self.failed_images.add(str(image_path))
                    return False
                    
            return result
        except Exception as e:
            logging.error(f"Error procesando {image_path}: {e}")
            self.failed_images.add(str(image_path))
            return False

    def copy_image(self, image_path):
        """Copia una imagen sin modificarla a la carpeta destino"""
        try:
            # Si la imagen ya fue procesada exitosamente, saltar
            if str(image_path) in self.processed_images:
                return True
            
            relative_path = image_path.relative_to(self.source_folder)
            new_path = self.dest_folder / relative_path

            # Crear directorios si no existen
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # Copiar la imagen
            shutil.copy2(image_path, new_path)
            logging.info(f"Copiada sin cambios: {relative_path}")
            self.processed_images.add(str(image_path))
            return True
        except Exception as e:
            logging.error(f"Error copiando {image_path}: {e}")
            self.failed_images.add(str(image_path))
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
                else:
                    # La imagen no necesita ser redimensionada, copiarla tal cual
                    self.copy_image(image_path)
                    return True
        except Exception as e:
            logging.error(f"Error CPU procesando {image_path}: {e}")
            return False

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

            img = self.find_large_image(self.source_folder)

            if not img:
                print("*****************************************")
                print("No se encontraron imágenes para procesar.")
                print("*****************************************")
                logging.info("No se encontraron imágenes para procesar.")
                return

            image_paths = []
            for root, _, files in os.walk(self.source_folder):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")):
                        image_paths.append(Path(root) / file)

            if not image_paths:
                print("*****************************************")
                print("No se encontraron imágenes para procesar.")
                print("*****************************************")
                logging.info("No se encontraron imágenes para procesar.")
                return

            total_images = len(image_paths)
            print(f"Total de imágenes encontradas: {total_images}")
            logging.info(f"Total de imágenes encontradas: {total_images}")

            # Determinar el intervalo de progreso en base al 10% del total
            progress_interval = max(1, total_images // 10)

            start_processing_time = time.time()

            if USE_MULTITHREAD:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import multiprocessing
                max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
                print(f"Procesando imágenes en paralelo con {max_workers} hilos...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_path = {executor.submit(self.reduce_image, image_path): image_path for image_path in image_paths}
                    for i, future in enumerate(as_completed(future_to_path), start=1):
                        image_path = future_to_path[future]
                        try:
                            result = future.result()
                            if not result:
                                self.copy_image(image_path)
                        except Exception as exc:
                            logging.error(f"Error en hilo procesando {image_path}: {exc}")
                            self.copy_image(image_path)
                        # Mostrar progreso cada 10% del total
                        if i % progress_interval == 0 or i == total_images:
                            progress_percentage = (i / total_images) * 100
                            print(f"Progreso: {progress_percentage:.2f}% ({i}/{total_images})")
            else:
                # Procesamiento secuencial (un solo hilo)
                for index, image_path in enumerate(image_paths, start=1):
                    if not self.reduce_image(image_path):
                        self.copy_image(image_path)
                    # Mostrar progreso cada 10% del total
                    if index % progress_interval == 0 or index == total_images:
                        progress_percentage = (index / total_images) * 100
                        print(f"Progreso: {progress_percentage:.2f}% ({index}/{total_images})")

            end_processing_time = time.time() - start_processing_time

            print(f"Tiempo total procesando imágenes: {end_processing_time:.2f} segundos")
            logging.info(f"Tiempo total procesando imágenes: {end_processing_time:.2f} segundos")

            # Verificar imágenes no procesadas
            self.verify_processed_images(image_paths)

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

    def verify_processed_images(self, image_paths):
        """Verifica que todas las imágenes hayan sido procesadas correctamente"""
        not_processed = []
        for path in image_paths:
            relative_path = path.relative_to(self.source_folder)
            dest_path = self.dest_folder / relative_path
            
            if not dest_path.exists():
                not_processed.append(path)
                logging.warning(f"Imagen no procesada: {path}")
        
        if not_processed:
            print(f"ADVERTENCIA: {len(not_processed)} imágenes no fueron procesadas correctamente")
            logging.warning(f"{len(not_processed)} imágenes no fueron procesadas correctamente")
            
            # Último intento de procesar imágenes faltantes
            print("Intentando último procesamiento de imágenes faltantes...")
            for path in not_processed:
                print(f"Último intento para: {path}")
                self.copy_image(path)
        else:
            print("Todas las imágenes fueron procesadas correctamente")
            logging.info("Todas las imágenes fueron procesadas correctamente")

    def compare_folders(self):
        """Compara las carpetas origen y destino para verificar integridad"""
        try:
            origin_files = set()
            for root, _, files in os.walk(self.source_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                        rel_path = os.path.relpath(os.path.join(root, file), self.source_folder)
                        origin_files.add(rel_path)
            
            dest_files = set()
            for root, _, files in os.walk(self.dest_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                        rel_path = os.path.relpath(os.path.join(root, file), self.dest_folder)
                        dest_files.add(rel_path)

            comparison_info = f"Comparación de carpetas:\n"
            comparison_info += f"Archivos de imagen en origen: {len(origin_files)}\n"
            comparison_info += f"Archivos de imagen en destino: {len(dest_files)}\n"

            missing_files = origin_files - dest_files
            if missing_files:
                comparison_info += f"¡ADVERTENCIA! Faltan {len(missing_files)} archivos en el destino:\n"
                for f in missing_files:
                    comparison_info += f"  - {f}\n"
            else:
                comparison_info += "Todos los archivos de imagen están presentes en el destino\n"

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

    try:
        if len(sys.argv) > 1:
            source_folder = sys.argv[1]
            print(f"Carpeta: {source_folder}")

            if not os.path.exists(source_folder):
                print(f"Error: La carpeta {source_folder} no existe.")
                sys.exit(1)
            
            image_reducer = ImageReducer(source_folder)
            print("Instancia de ImageReducer creada")

            image_reducer.process_folder()
            print("Procesamiento de carpeta completado")
        else:
            print("Por favor, elige una o más carpetas y usa la opción de menú contextual \"Reducir imagenes\".")
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
        logging.info("Proceso interrumpido por el usuario.")
        sys.exit(130)