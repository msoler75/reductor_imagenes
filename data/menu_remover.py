import os
import sys
import winreg
import traceback
import ctypes

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

def remove_context_menu():
    """Elimina la entrada del menú contextual de Windows"""
    try:
        # Verificar si la clave existe antes de intentar eliminarla
        try:
            key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Directory\shell\ReducirImagenes")
            winreg.CloseKey(key)
            key_exists = True
        except FileNotFoundError:
            key_exists = False
        
        if not key_exists:
            print("La entrada de menú contextual no está instalada.")
            return True
        
        # Primero eliminar la subclave 'command'
        try:
            winreg.DeleteKey(winreg.HKEY_CLASSES_ROOT, r"Directory\shell\ReducirImagenes\command")
            print("Subclave 'command' eliminada correctamente.")
        except Exception as e:
            print(f"Error al eliminar subclave 'command': {e}")
        
        # Luego eliminar la clave principal
        try:
            winreg.DeleteKey(winreg.HKEY_CLASSES_ROOT, r"Directory\shell\ReducirImagenes")
            print("Clave principal eliminada correctamente.")
        except Exception as e:
            print(f"Error al eliminar clave principal: {e}")
            return False
        
        print("✅ Entrada de menú contextual eliminada exitosamente.")
        return True
        
    except Exception as e:
        print(f"❌ Error eliminando menú contextual: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("==============================================")
    print("  DESINSTALACIÓN DE MENÚ CONTEXTUAL DE WINDOWS")
    print("==============================================")
    print("")
    
    # Verificar si se ejecuta como administrador
    if not is_admin():
        print("Esta operación requiere permisos de administrador.")
        if run_as_admin():
            print("Por favor, continúe en la nueva ventana elevada.")
        else:
            print("No se pudo obtener permisos de administrador. La desinstalación no puede continuar.")
        input("Presione ENTER para salir...")
        sys.exit(0)
    
    try:
        if remove_context_menu():
            print("Desinstalación completada correctamente.")
        else:
            print("La desinstalación no se completó correctamente.")
            
        print("")
        print("==============================================")
        input("Presione ENTER para continuar...")
    except Exception as e:
        print(f"Error general durante la desinstalación: {e}")
        traceback.print_exc()
        input("Presione ENTER para salir...")