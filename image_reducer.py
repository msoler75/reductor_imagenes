import os
import sys
import logging
import shutil
from pathlib import Path
from PIL import Image, ImageFilter
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QMessageBox
from PyQt5.QtGui import QIcon
import winreg
import ctypes

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
        folder_name = self.source_folder.name
        
        # Modificar el nombre de la carpeta de destino según las reglas
        if folder_name.endswith("-"):
            dest_folder_name = f"{folder_name.rstrip('-')} g -"
        elif "- g -" in folder_name:
            dest_folder_name = f"{folder_name} g -"
        else:
            dest_folder_name = f"{folder_name} - g -"
        
        self.dest_folder = self.source_folder.parent / dest_folder_name
        self.log_file = self.dest_folder / "image_reduction_log.txt"
        
        # Configurar logging
        os.makedirs(self.dest_folder, exist_ok=True)
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')
        
        self.total_processed = 0
        self.total_reduced = 0
        self.total_copied = 0
        self.errors = 0

    def reduce_image(self, image_path):
        """Redimensiona una imagen si es necesario"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Condición para redimensionar
                if width > 1920 or height > 1920:
                    # Mantener proporción usando el método actualizado
                    img.thumbnail((1920, 1920), RESAMPLING_METHOD)
                    
                    # Guardar en nueva ubicación
                    relative_path = image_path.relative_to(self.source_folder)
                    new_path = self.dest_folder / relative_path
                    
                    # Crear directorios si no existen
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    img.save(new_path)
                    logging.info(f"Redimensionada: {relative_path}")
                    return True
                return False
        except Exception as e:
            logging.error(f"Error procesando {image_path}: {e}")
            return False

    def process_folder(self):
        """Procesa recursivamente la carpeta"""
        print(f"Iniciando procesamiento de {self.source_folder}")
        logging.info(f"Iniciando procesamiento de {self.source_folder}")
        
        try:
            # Contar el número total de archivos
            total_files = sum([len(files) for _, _, files in os.walk(self.source_folder)])
            
            for root, _, files in os.walk(self.source_folder):
                for file in files:
                    self.total_processed += 1
                    
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.source_folder)
                    dest_path = self.dest_folder / relative_path
                    
                    # Calcular y mostrar el porcentaje de progreso
                    progress = (self.total_processed / total_files) * 100
                    print(f"Progreso: {progress:.2f}% ({self.total_processed}/{total_files})")
                    
                    # Crear directorio destino
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Verificar si es imagen
                    try:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                            if self.reduce_image(file_path):
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
        except Exception as e:
            logging.error(f"Error general: {e}")
            print(f"Error: {e}")
        
        self.show_summary()
        logging.info("Procesamiento finalizado")
        print("========================")
        print("Procesamiento finalizado")
        print("========================")

    def show_summary(self):
        """Muestra resumen en consola y notificación"""
        summary = (f"\nResumen del proceso:\n"
                   f"Total procesados: {self.total_processed}\n"
                   f"Imágenes reducidas: {self.total_reduced}\n"
                   f"Archivos copiados: {self.total_copied}\n"
                   f"Errores: {self.errors}")
        
        print(summary)
        logging.info(summary)
        
        # Mostrar notificación
        try:
            app = QApplication(sys.argv)
            tray_icon = QSystemTrayIcon()
            
            message = (f"Reducción de imágenes completada:\n"
                      f"Total procesados: {self.total_processed}\n"
                      f"Imágenes reducidas: {self.total_reduced}")
            
            tray_icon.showMessage("Reducción de Imágenes", message)
            app.processEvents()  # Procesar eventos sin bloquear
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


def main():
   
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
        if opcion == 'A' or opcion == 'a':
            add_context_menu()
            input("Presione Enter para salir...")
        elif opcion == 'B' or opcion == 'b':
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