import os
import sys
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageFilter
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QMessageBox
from PyQt5.QtGui import QIcon
import winreg
import ctypes
import filecmp
import difflib
from concurrent.futures import ThreadPoolExecutor
import gc

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
                
            # Si tiene imágenes reducidas, añadir "REDUCIDAS"
            dest_folder_name = f"{dest_folder_name} REDUCIDAS"
        
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

    def process_folder(self):
        """Procesa recursivamente la carpeta"""
        print(f"Analizando carpeta {self.source_folder}...")
        logging.info(f"Iniciando procesamiento de {self.source_folder}")
        
        # Verificar si realmente es necesario procesar
        if not self.needs_processing:
            message = f"No hay imágenes grandes para reducir en {self.source_folder}"
            print(message)
            logging.info(message)
            self.show_notification("Análisis Completado", message)
            return
        
        try:
            # Contar el número total de archivos
            total_files = sum([len(files) for _, _, files in os.walk(self.source_folder)])
            
            print(f"Iniciando reducción de imágenes en {self.source_folder}")
            print(f"Total de archivos a procesar: {total_files}")
            
            # Usar multithreading para mejorar rendimiento
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for root, dirs, files in os.walk(self.source_folder):
                    # Procesar la carpeta actual
                    current_folder = Path(root)
                    relative_folder = current_folder.relative_to(self.source_folder) if current_folder != self.source_folder else Path("")
                    dest_folder = self.dest_folder / relative_folder
                    
                    # Crear directorio destino
                    dest_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Registrar carpeta procesada
                    self.processed_folders.append(str(relative_folder))
                    
                    # Procesar archivos
                    for file in files:
                        self.total_processed += 1
                        
                        file_path = current_folder / file
                        dest_path = dest_folder / file
                        
                        # Calcular y mostrar el porcentaje de progreso
                        progress = (self.total_processed / total_files) * 100
                        if self.total_processed % 10 == 0 or self.total_processed == total_files:
                            print(f"Progreso: {progress:.2f}% ({self.total_processed}/{total_files})")
                        
                        # Verificar si es imagen
                        try:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                                # Usar executor para procesar imágenes en paralelo
                                future = executor.submit(self.reduce_image, file_path)
                                if future.result():
                                    self.total_reduced += 1
                                else:
                                    # Copiar imagen sin cambios
                                    shutil.copy2(file_path, dest_path)
                                    self.total_copied += 1
                            else:
                                # Copiar archivos que no son imágenes
                                shutil.copy2(file_path, dest_path)
                                self.total_copied += 1
                        except Exception as e:
                            logging.error(f"Error procesando {file_path}: {e}")
                            self.errors += 1
            
            # Comparar carpetas después del procesamiento
            self.compare_folders()
            
            # Actualizar registro de procesamiento
            self.update_registry()
            
        except Exception as e:
            logging.error(f"Error general: {e}")
            print(f"Error: {e}")
        
        self.show_summary()
        logging.info("Procesamiento finalizado")
        print("========================")
        print("Procesamiento finalizado")
        print("========================")

    def compare_folders(self):
        """Compara las carpetas origen y destino para verificar integridad"""
        try:
            origin_file_count = sum([len(files) for _, _, files in os.walk(self.source_folder)])
            dest_file_count = sum([len(files) for _, _, files in os.walk(self.dest_folder)])
            
            comparison_info = f"Comparación de carpetas:\n"
            comparison_info += f"Archivos en origen: {origin_file_count}\n"
            comparison_info += f"Archivos en destino: {dest_file_count}\n"
            
            if origin_file_count != dest_file_count:
                comparison_info += "¡ADVERTENCIA! El número de archivos no coincide\n"
            else:
                comparison_info += "El número de archivos coincide correctamente\n"
            
            # Verificar contenido de algunos archivos al azar como muestra
            different_files = []
            
            for root, _, files in os.walk(self.source_folder):
                if len(files) > 0:
                    # Tomar una muestra de archivos no-imágenes para comparar
                    sample_files = [f for f in files if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
                    sample_size = min(5, len(sample_files))
                    
                    for i in range(sample_size):
                        if i < len(sample_files):
                            file = sample_files[i]
                            src_path = Path(root) / file
                            rel_path = src_path.relative_to(self.source_folder)
                            dst_path = self.dest_folder / rel_path
                            
                            if dst_path.exists() and not filecmp.cmp(src_path, dst_path, shallow=False):
                                different_files.append(str(rel_path))
            
            if different_files:
                comparison_info += f"Se detectaron {len(different_files)} archivos con contenido diferente\n"
                self.different_files = different_files
            else:
                comparison_info += "No se detectaron diferencias en los archivos de muestra\n"
            
            logging.info(comparison_info)
            print(comparison_info)
            
        except Exception as e:
            logging.error(f"Error comparando carpetas: {e}")
            print(f"Error comparando carpetas: {e}")

    def update_registry(self):
        """Actualiza el registro de carpetas procesadas"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.processed_registry, "a", encoding="utf-8") as f:
                f.write(f"\n=== Procesamiento del {timestamp} ===\n")
                f.write(f"Carpeta origen: {self.source_folder}\n")
                f.write(f"Carpeta destino: {self.dest_folder}\n")
                f.write(f"Total archivos: {self.total_processed}\n")
                f.write(f"Imágenes reducidas: {self.total_reduced}\n")
                f.write(f"Archivos copiados sin cambios: {self.total_copied}\n")
                f.write(f"Errores: {self.errors}\n")
                
                f.write("\nCarpetas procesadas:\n")
                for folder in self.processed_folders:
                    f.write(f"- {folder}\n")
                
                if self.different_files:
                    f.write("\nArchivos con diferencias (muestra):\n")
                    for diff_file in self.different_files:
                        f.write(f"- {diff_file}\n")
                
                f.write("\n")
            
            logging.info(f"Registro actualizado: {self.processed_registry}")
        except Exception as e:
            logging.error(f"Error actualizando registro: {e}")

    def show_summary(self):
        """Muestra resumen en consola y notificación"""
        summary = (f"\nResumen del proceso:\n"
                   f"Total procesados: {self.total_processed}\n"
                   f"Imágenes reducidas: {self.total_reduced}\n"
                   f"Archivos copiados: {self.total_copied}\n"
                   f"Errores: {self.errors}\n"
                   f"Carpeta destino: {self.dest_folder}")
        
        print(summary)
        logging.info(summary)
        
        # Mostrar notificación
        message = (f"Reducción de imágenes completada:\n"
                  f"Total procesados: {self.total_processed}\n"
                  f"Imágenes reducidas: {self.total_reduced}")
        
        self.show_notification("Reducción Completada", message)

    def show_notification(self, title, message):
        """Muestra una notificación en el system tray"""
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            tray_icon = QSystemTrayIcon()
            tray_icon.setIcon(QIcon())
            tray_icon.setVisible(True)
            
            # Mostrar notificación
            tray_icon.showMessage(title, message, QSystemTrayIcon.Information, 5000)
            
            # Esperar un poco para que se muestre la notificación
            for _ in range(10):
                app.processEvents()
                time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"Error en notificación: {e}")
            print(f"Error en notificación: {e}")


def is_admin():
    """Verificar si el programa se está ejecutando con permisos de administrador"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    """Relanzar el script con permisos de administrador"""
    # Obtener la ruta del script actual
    if sys.argv[0].endswith('.py'):
        # Si se ejecuta directamente desde el script .py
        executable = sys.executable
        script = sys.argv[0]
    else:
        # Si es un ejecutable compilado
        executable = sys.argv[0]
        script = sys.argv[0]
    
    # Intentar elevar permisos usando ShellExecute
    try:
        # Pasar argumentos originales si los hay
        args = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else ''
        
        # Usar ShellExecute para solicitar elevación
        ctypes.windll.shell32.ShellExecuteW(
            None, 
            "runas", 
            executable, 
            f'"{script}" {args}', 
            None, 
            1
        )
        # Salir del proceso actual
        sys.exit(0)
    except Exception as e:
        print(f"No se pudo elevar permisos: {e}")
        input("Presione Enter para salir...")
        sys.exit(1)

def add_context_menu():
    """Añade entrada al menú contextual de Windows"""
    try:
        key_path = r"Directory\shell\ReducirImagenes"
        command_path = r"Directory\shell\ReducirImagenes\command"
        
        # Obtener la ruta de la carpeta de instalación
        install_dir = os.path.join(os.path.expanduser("~"), "ReduccionImagenes")
        icon_path = os.path.join(install_dir, "icono.ico")  # Ruta al icono en la carpeta de instalación
        
        # Crear clave para menú contextual
        with winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, key_path) as key:
            winreg.SetValue(key, "", winreg.REG_SZ, "Reducir Imágenes")
            winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, icon_path)  # añade el icono
        
        # Establecer comando
        python_executable = sys.executable
        script_path = os.path.abspath(__file__)
        
        with winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, command_path) as key:
            command = f'"{python_executable}" "{script_path}" "%1"'
            winreg.SetValue(key, "", winreg.REG_SZ, command)
        
        print("Entrada de menú contextual añadida exitosamente.")
    except Exception as e:
        print(f"Error añadiendo menú contextual: {e}")

def remove_context_menu():
    """Elimina la entrada del menú contextual de Windows"""
    try:
        winreg.DeleteKey(winreg.HKEY_CLASSES_ROOT, r"Directory\shell\ReducirImagenes\command")
        winreg.DeleteKey(winreg.HKEY_CLASSES_ROOT, r"Directory\shell\ReducirImagenes")
        print("Entrada de menú contextual eliminada exitosamente.")
    except Exception as e:
        print(f"Error eliminando menú contextual: {e}")

def show_gpu_info():
    """Muestra información sobre la GPU si está disponible"""
    if HAS_CUDA:
        try:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Desconocido"
            
            print(f"Aceleración GPU disponible: {HAS_CUDA}")
            print(f"Número de GPUs: {gpu_count}")
            print(f"GPU principal: {gpu_name}")
            return True
        except Exception as e:
            print(f"Error obteniendo información GPU: {e}")
    else:
        print("Aceleración GPU no disponible. Se usará CPU.")
    return False

def main():
    print("====================================")
    print("Reductor de Imágenes v2.0")
    print("====================================")
    
    # Verificar disponibilidad de GPU
    has_gpu = show_gpu_info()
    print("====================================")
   
    if len(sys.argv) > 1:
        source_folder = sys.argv[1]
        try:
            reducer = ImageReducer(source_folder)
            reducer.process_folder()
        except Exception as e:
            print(f"Error general: {e}")
            logging.error(f"Error general: {e}")
        finally:
            input("Presione Enter para salir...")
    else:
        # Si no es administrador, solicitar elevación
        if not is_admin():
            print("Se requieren permisos de administrador.")
            print("Reiniciando en modo administrador...")
            run_as_admin()
            return    

        # Si no hay argumentos, mostrar menú de instalación
        print("Opciones:")
        print("A. Instalar menú contextual")
        print("B. Desinstalar menú contextual")
        print("0. Salir")
        
        opcion = input("Seleccione una opción (A/B/0): ")
        if opcion.upper() == 'A':
            add_context_menu()
            input("Presione Enter para salir...")
        elif opcion.upper() == 'B':
            remove_context_menu()
            input("Presione Enter para salir...")
        elif opcion == '0':
            print("Saliendo...")
            input("Presione Enter para salir...")
            exit()
        else:
            print("Opción inválida")
            input("Presione Enter para salir...")

if __name__ == "__main__":
    main()