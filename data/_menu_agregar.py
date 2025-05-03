import os
import sys
import winreg
import traceback
import ctypes
import argparse

def is_admin():
    """Verificar si el programa se está ejecutando con permisos de administrador"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    """Relanzar el script con permisos de administrador"""
    try:
        # Si es un archivo Python
        if sys.argv[0].endswith('.py'):
            executable = sys.executable
            script = os.path.abspath(sys.argv[0])
            args = [executable, script] + sys.argv[1:]
        else:
            # Si es un ejecutable compilado
            args = [os.path.abspath(sys.argv[0])] + sys.argv[1:]
        
        print("Solicitando permisos de administrador...")
        
        # Usar ShellExecute para solicitar elevación
        ctypes.windll.shell32.ShellExecuteW(
            None, 
            "runas", 
            args[0], 
            " ".join(f'"{arg}"' for arg in args[1:]), 
            None, 
            1
        )
        return True
    except Exception as e:
        print(f"Error al solicitar permisos de administrador: {e}")
        traceback.print_exc()
        return False

def add_context_menu():
    """Añade entrada al menú contextual de Windows"""
    try:
        key_path = r"Directory\shell\ReducirImagenes"
        command_path = r"Directory\shell\ReducirImagenes\command"
        
        # Determinar la ruta correcta del ejecutable/script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cmd_path = os.path.join(script_dir, "iniciar_reductor.cmd")
        
        # Asegurar que la ruta al CMD existe
        if not os.path.exists(cmd_path):
            print(f"Error: No se encuentra el archivo {cmd_path}")
            return False
        
        # Ajuste para manejar correctamente los argumentos con espacios
        command = f'cmd.exe /c {cmd_path} "%1\"'
        
        # Verificar si existe un icono en la carpeta de instalación
        icon_path = os.path.join(script_dir, "icono.ico")
        
        # Verificar si el icono existe
        if not os.path.exists(icon_path):
            print(f"Advertencia: El icono no existe en {icon_path}")
            print("Se usará el icono predeterminado.")
            icon_param = ""
        else:
            icon_param = f',"{icon_path}"'
        
        # Crear clave para menú contextual
        with winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, key_path) as key:
            winreg.SetValue(key, "", winreg.REG_SZ, "Reducir Imágenes")
            if os.path.exists(icon_path):
                winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, icon_path)
        
        # Establecer comando
        with winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, command_path) as key:
            winreg.SetValue(key, "", winreg.REG_SZ, command)
        
        print("✅ Entrada de menú contextual añadida exitosamente.")
        print("Ahora puedes hacer clic derecho en cualquier carpeta y seleccionar 'Reducir Imágenes'")
        return True
        
    except Exception as e:
        print(f"❌ Error añadiendo menú contextual: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("==============================================")
    print("   INSTALACIÓN DE MENÚ CONTEXTUAL DE WINDOWS")
    print("==============================================")
    print("")
    
    # Verificar si se ejecuta como administrador
    if not is_admin():
        print("Esta operación requiere permisos de administrador.")
        if run_as_admin():
            print("Por favor, continúe en la nueva ventana elevada.")
        else:
            print("No se pudo obtener permisos de administrador. La instalación no puede continuar.")
        input("Presione ENTER para salir...")
        sys.exit(0)
    
    try:
        if add_context_menu():
            print("Instalación completada correctamente.")
        else:
            print("La instalación no se completó correctamente.")
            
        print("")
        print("==============================================")
        input("Presione ENTER para continuar...")
    except Exception as e:
        print(f"Error general durante la instalación: {e}")
        traceback.print_exc()
        input("Presione ENTER para salir...")