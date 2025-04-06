import sys
import torch

print("Iniciando prueba de CUDA...")
sys.stdout.flush()

try:
    print("Versión de PyTorch:", torch.__version__)
    sys.stdout.flush()
    print("Verificando CUDA...")
    sys.stdout.flush()
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible: {cuda_available}")
    sys.stdout.flush()
except Exception as e:
    print(f"Error detectado: {type(e).__name__}: {e}")
    sys.stdout.flush()

print("Prueba completada")
sys.stdout.flush()
input("Presione Enter para salir...")