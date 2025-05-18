# Benchmarks 
# 6. ArrayFire
# 
# Libraries: ArrayFire Python bindings
# Advantages: High-performance library with image processing capabilities
# Notes: Supports multiple backends (CUDA, OpenCL, CPU)
# 
# 7. PyCUDA
# 
# Libraries: PyCUDA
# Advantages: Direct Python interface to CUDA, allows custom kernels
# Notes: More control but requires CUDA programming knowledge
# 
# Libraries: Numba with CUDA support
# Advantages: JIT compilation of Python to GPU code
# Notes: Allows Python functions to run on GPU with minimal changes
# 
# 9. GPU-accelerated FFT libraries
# 
# Libraries: cuFFT (via PyCUDA/CuPy)
# Advantages: Very fast for frequency domain image resizing
# Notes: Different approach using frequency domain transformation


import os
import time
from PIL import Image
import numpy as np
from pathlib import Path

# PyTorch imports
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# OpenCV y CUDA imports
import cv2
import cuda
import tensorrt as trt

# CuPy import
import cupy as cp

# ArrayFire import
import arrayfire as af

# PyCUDA imports
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray

# Configuración general
BATCH_SIZE = 32
TARGET_SIZE = (1920, 1920)

# Definir las rutas de las carpetas
source_folder = Path('input_images')  # Carpeta con las imágenes originales
output_base = Path('output')  # Carpeta base para los resultados

# Crear carpetas de salida para cada método
methods = ['PyTorch', 'PyTorch_Batch', 
          'TensorFlow', 'TensorFlow_Batch',
          'OpenCV_CUDA', 'OpenCV_CUDA_Batch',
          'CuPy', 'CuPy_Batch',
          'ArrayFire', 'ArrayFire_Batch',
          'PyCUDA', 'PyCUDA_Batch']

for method in methods:
    (output_base / method).mkdir(parents=True, exist_ok=True)

# Clase auxiliar para cargar imágenes en lotes
class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.image_files = [f for f in self.folder_path.glob('*.jpg') or self.folder_path.glob('*.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, str(img_path.name)

# Función para medir tiempo
def timer_func(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'{func.__name__}: {end - start:.4f} segundos')
        return result
    return wrapper

# Definir las transformaciones para PyTorch y TensorFlow/Keras
torch_transform = transforms.Compose([
    transforms.Resize((1920, 1920)),
])

tensorflow_transform = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resize(1920, 1920),
])

# Definir la función para hacer benchmarking
def benchmark(method_name, transform):
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Aplicar la transformación
            if method_name == 'PyTorch':
                img = torch_transform(img)
            elif method_name == 'TensorFlow':
                img = tf.expand_dims(load_img(os.path.join(source_folder, filename)), 0)
                img = tensorflow_transform(img)
                img = tf.squeeze(img, [0])
            
            # Guardar la imagen en una nueva carpeta
            img.save(os.path.join(source_folder, method_name, filename))
    
    end = time.time()
    print(f'{method_name}: {end - start} segundos')


# Definir la función para hacer benchmarking con NVIDIA NPP
def benchmark_npp():
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(source_folder, filename))  # Usamos OpenCV para leer la imagen
            
            # Aplicar transformación de redimensionamiento con NPP (NVIDIA Performance Primitives)
            dst = cv2.resize(img, (1920, 1920), interpolation=cv2.INTER_LINEAR)
            
            # Guardar la imagen en una nueva carpeta
            cv2.imwrite(os.path.join(source_folder, 'NVIDIA NPP', filename), dst)
    
    end = time.time()
    print('NVIDIA NPP: {} segundos'.format(end - start))



# Definir la función para hacer benchmarking con OpenCV con CUDA
def benchmark_opencv_cuda():
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(source_folder, filename))   # Usamos OpenCV para leer la imagen
            
            # Aplicar transformación de redimensionamiento con OpenCV (con soporte para CUDA)
            dst = cv2.cuda.resize(img, (1920, 1920))
            
            # Guardar la imagen en una nueva carpeta
            cv2.imwrite(os.path.join(source_folder, 'OpenCV with CUDA', filename), dst)
    
    end = time.time()
    print('OpenCV con CUDA: {} segundos'.format(end - start))

# Definir la función para hacer benchmarking con CuPy
def benchmark_cupy():
    import cupy as cp
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Convertir la imagen a un array de CuPy
            img_array = cp.array(img)
            
            # Aplicar transformación de redimensionamiento con CuPy
            img_resized = cp.resize(img_array, (1920, 1920))
            
            # Guardar la imagen en una nueva carpeta
            img_resized_pil = Image.fromarray(cp.asnumpy(img_resized))
            img_resized_pil.save(os.path.join(source_folder, 'CuPy', filename))
    
    end = time.time()
    print('CuPy: {} segundos'.format(end - start))

# Definir la función para hacer benchmarking con CuPy y TensorRT
def benchmark_cupy_tensorrt():
    import cupy as cp
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Convertir la imagen a un array de CuPy
            img_array = cp.array(img)
            
            # Aplicar transformación de redimensionamiento con Cu


def benchmark_tensorrt():
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Convertir la imagen a un array de CuPy
            img_array = cp.array(img)
            
            # Aplicar transformación de redimensionamiento con TensorRT (opcional, requiere configuración adicional)
            # Aquí se debe implementar el código específico para TensorRT
            
            # Guardar la imagen en una nueva carpeta
            img_resized_pil = Image.fromarray(cp.asnumpy(img_array))
            img_resized_pil.save(os.path.join(source_folder, 'TensorRT', filename))
    
    end = time.time()
    print('TensorRT: {} segundos'.format(end - start))

# Definir la función para hacer benchmarking con ArrayFire
def benchmark_arrayfire():
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Convertir la imagen a un array de ArrayFire
            img_array = af.to_array(img)
            
            # Aplicar transformación de redimensionamiento con ArrayFire
            img_resized = af.resize(img_array, (1920, 1920))
            
            # Guardar la imagen en una nueva carpeta
            img_resized_pil = Image.fromarray(af.to_numpy(img_resized))
            img_resized_pil.save(os.path.join(source_folder, 'ArrayFire', filename))
    
    end = time.time()
    print('ArrayFire: {} segundos'.format(end - start))


# Definir la función para hacer benchmarking con PyCUDA
def benchmark_pycuda():
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Convertir la imagen a un array de PyCUDA
            img_array = cp.array(img)
            
            # Aplicar transformación de redimensionamiento con PyCUDA (opcional, requiere configuración adicional)
            # Aquí se debe implementar el código específico para PyCUDA
            
            # Guardar la imagen en una nueva carpeta
            img_resized_pil = Image.fromarray(cp.asnumpy(img_array))
            img_resized_pil.save(os.path.join(source_folder, 'PyCUDA', filename))
    
    end = time.time()
    print('PyCUDA: {} segundos'.format(end - start))

@timer_func
def benchmark_pytorch_batch():
    # Configurar el dispositivo CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear el dataset y dataloader
    dataset = ImageDataset(source_folder, transform=torch_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    for batch_images, batch_filenames in dataloader:
        batch_images = batch_images.to(device)
        resized_images = torch_transform(batch_images)
        
        # Guardar las imágenes procesadas
        for img, filename in zip(resized_images.cpu(), batch_filenames):
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(output_base / 'PyTorch_Batch' / filename)

@timer_func
def benchmark_tensorflow_batch():
    dataset = tf.data.Dataset.from_tensor_slices(list(source_folder.glob('*.jpg')))
    dataset = dataset.map(lambda x: tf.io.read_file(x))
    dataset = dataset.map(lambda x: tf.io.decode_jpeg(x, channels=3))
    dataset = dataset.batch(BATCH_SIZE)

    for batch in dataset:
        resized = tensorflow_transform(batch)
        for i, img in enumerate(resized):
            tf.keras.preprocessing.image.save_img(
                output_base / 'TensorFlow_Batch' / f'img_{i}.jpg', 
                img.numpy()
            )

@timer_func
def benchmark_opencv_cuda_batch():
    # Corregir el método anterior que usaba cuda_FHDML
    images = []
    filenames = []
    
    # Recopilar imágenes para el procesamiento por lotes
    for file in source_folder.glob('*.jpg'):
        img = cv2.imread(str(file))
        if img is not None:
            images.append(img)
            filenames.append(file.name)
            
            if len(images) == BATCH_SIZE:
                # Crear stream CUDA
                stream = cv2.cuda_Stream()
                
                # Subir lote a GPU
                gpu_images = [cv2.cuda.GpuMat(img) for img in images]
                
                # Redimensionar en paralelo
                resized = [cv2.cuda.resize(img, TARGET_SIZE) for img in gpu_images]
                
                # Descargar y guardar
                for img, filename in zip(resized, filenames):
                    result = img.download()
                    cv2.imwrite(str(output_base / 'OpenCV_CUDA_Batch' / filename), result)
                
                images = []
                filenames = []
    
    # Procesar las imágenes restantes
    if images:
        stream = cv2.cuda_Stream()
        gpu_images = [cv2.cuda.GpuMat(img) for img in images]
        resized = [cv2.cuda.resize(img, TARGET_SIZE) for img in gpu_images]
        for img, filename in zip(resized, filenames):
            result = img.download()
            cv2.imwrite(str(output_base / 'OpenCV_CUDA_Batch' / filename), result)

@timer_func
def benchmark_cupy_batch():
    images = []
    filenames = []
    
    for file in source_folder.glob('*.jpg'):
        img = cp.array(Image.open(file))
        images.append(img)
        filenames.append(file.name)
        
        if len(images) == BATCH_SIZE:
            # Procesar lote
            batch = cp.stack(images)
            resized = cp.ndarray((len(batch), *TARGET_SIZE, 3))
            
            for i, img in enumerate(batch):
                resized[i] = cp.resize(img, TARGET_SIZE)
            
            # Guardar resultados
            for img, filename in zip(resized, filenames):
                img_pil = Image.fromarray(cp.asnumpy(img).astype(np.uint8))
                img_pil.save(output_base / 'CuPy_Batch' / filename)
            
            images = []
            filenames = []
    
    # Procesar imágenes restantes
    if images:
        batch = cp.stack(images)
        resized = cp.ndarray((len(batch), *TARGET_SIZE, 3))
        for i, img in enumerate(batch):
            resized[i] = cp.resize(img, TARGET_SIZE)
        
        for img, filename in zip(resized, filenames):
            img_pil = Image.fromarray(cp.asnumpy(img).astype(np.uint8))
            img_pil.save(output_base / 'CuPy_Batch' / filename)

@timer_func
def benchmark_arrayfire_batch():
    images = []
    filenames = []
    
    for file in source_folder.glob('*.jpg'):
        img = af.to_array(np.array(Image.open(file)))
        images.append(img)
        filenames.append(file.name)
        
        if len(images) == BATCH_SIZE:
            # Procesar lote
            batch = af.join(0, *images)  # Unir imágenes en un solo tensor
            resized = af.resize(batch, *TARGET_SIZE)
            
            # Guardar resultados
            for i, filename in enumerate(filenames):
                img = af.to_array(resized[i])
                img_pil = Image.fromarray(img.to_numpy().astype(np.uint8))
                img_pil.save(output_base / 'ArrayFire_Batch' / filename)
            
            images = []
            filenames = []
    
    # Procesar imágenes restantes
    if images:
        batch = af.join(0, *images)
        resized = af.resize(batch, *TARGET_SIZE)
        
        for i, filename in enumerate(filenames):
            img = af.to_array(resized[i])
            img_pil = Image.fromarray(img.to_numpy().astype(np.uint8))
            img_pil.save(output_base / 'ArrayFire_Batch' / filename)

@timer_func
def benchmark_pycuda_batch():
    # Definir kernel CUDA para redimensionamiento
    kernel_code = """
    __global__ void resize_kernel(float *input, float *output, int in_w, int in_h, int out_w, int out_h) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < out_w * out_h) {
            int out_x = idx % out_w;
            int out_y = idx / out_w;
            
            float scale_x = float(in_w) / out_w;
            float scale_y = float(in_h) / out_h;
            
            int in_x = int(out_x * scale_x);
            int in_y = int(out_y * scale_y);
            
            output[idx] = input[in_y * in_w + in_x];
        }
    }
    """
    
    # Compilar el kernel
    mod = cuda.module_from_source(kernel_code)
    resize_kernel = mod.get_function("resize_kernel")
    
    images = []
    filenames = []
    
    for file in source_folder.glob('*.jpg'):
        img = np.array(Image.open(file)).astype(np.float32)
        images.append(img)
        filenames.append(file.name)
        
        if len(images) == BATCH_SIZE:
            # Procesar lote
            batch = np.stack(images)
            input_gpu = gpuarray.to_gpu(batch)
            output = np.empty((len(batch), *TARGET_SIZE, 3), dtype=np.float32)
            output_gpu = gpuarray.empty_like(output)
            
            # Ejecutar kernel para cada imagen en el lote
            block_size = (256, 1, 1)
            grid_size = ((TARGET_SIZE[0] * TARGET_SIZE[1] + block_size[0] - 1) // block_size[0], 1)
            
            for i in range(len(batch)):
                resize_kernel(
                    input_gpu[i].gpudata,
                    output_gpu[i].gpudata,
                    np.int32(batch.shape[2]),
                    np.int32(batch.shape[1]),
                    np.int32(TARGET_SIZE[0]),
                    np.int32(TARGET_SIZE[1]),
                    block=block_size,
                    grid=grid_size
                )
            
            # Guardar resultados
            output = output_gpu.get()
            for img, filename in zip(output, filenames):
                img_pil = Image.fromarray(img.astype(np.uint8))
                img_pil.save(output_base / 'PyCUDA_Batch' / filename)
            
            images = []
            filenames = []
    
    # Procesar imágenes restantes
    if images:
        batch = np.stack(images)
        input_gpu = gpuarray.to_gpu(batch)
        output = np.empty((len(batch), *TARGET_SIZE, 3), dtype=np.float32)
        output_gpu = gpuarray.empty_like(output)
        
        for i in range(len(batch)):
            resize_kernel(
                input_gpu[i].gpudata,
                output_gpu[i].gpudata,
                np.int32(batch.shape[2]),
                np.int32(batch.shape[1]),
                np.int32(TARGET_SIZE[0]),
                np.int32(TARGET_SIZE[1]),
                block=block_size,
                grid=grid_size
            )
        
        output = output_gpu.get()
        for img, filename in zip(output, filenames):
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_pil.save(output_base / 'PyCUDA_Batch' / filename)

if __name__ == '__main__':
    print("Ejecutando benchmarks...")
    
    # Ejecutar versiones single
    benchmark('PyTorch', torch_transform)
    benchmark('TensorFlow', tensorflow_transform)
    benchmark_opencv_cuda()
    benchmark_cupy()
    benchmark_arrayfire()
    benchmark_pycuda()
    
    print("\nEjecutando versiones batch...")
    
    # Ejecutar versiones batch
    benchmark_pytorch_batch()
    benchmark_tensorflow_batch()
    benchmark_opencv_cuda_batch()
    benchmark_cupy_batch()
    benchmark_arrayfire_batch()
    benchmark_pycuda_batch()
    
    print("\n¡Benchmarks completados!")
```
</copilot-edited-file>  # End of the edited file
```python
# Benchmarks 
# 6. ArrayFire
# 
# Libraries: ArrayFire Python bindings
# Advantages: High-performance library with image processing capabilities
# Notes: Supports multiple backends (CUDA, OpenCL, CPU)
# 
# 7. PyCUDA
# 
# Libraries: PyCUDA
# Advantages: Direct Python interface to CUDA, allows custom kernels
# Notes: More control but requires CUDA programming knowledge
# 
# Libraries: Numba with CUDA support
# Advantages: JIT compilation of Python to GPU code
# Notes: Allows Python functions to run on GPU with minimal changes
# 
# 9. GPU-accelerated FFT libraries
# 
# Libraries: cuFFT (via PyCUDA/CuPy)
# Advantages: Very fast for frequency domain image resizing
# Notes: Different approach using frequency domain transformation


import os
import time
from PIL import Image
import numpy as np
from pathlib import Path

# PyTorch imports
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# OpenCV y CUDA imports
import cv2
import cuda
import tensorrt as trt

# CuPy import
import cupy as cp

# ArrayFire import
import arrayfire as af

# PyCUDA imports
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray

# Configuración general
BATCH_SIZE = 32
TARGET_SIZE = (1920, 1920)

# Definir las rutas de las carpetas
source_folder = Path('input_images')  # Carpeta con las imágenes originales
output_base = Path('output')  # Carpeta base para los resultados

# Crear carpetas de salida para cada método
methods = ['PyTorch', 'PyTorch_Batch', 
          'TensorFlow', 'TensorFlow_Batch',
          'OpenCV_CUDA', 'OpenCV_CUDA_Batch',
          'CuPy', 'CuPy_Batch',
          'ArrayFire', 'ArrayFire_Batch',
          'PyCUDA', 'PyCUDA_Batch']

for method in methods:
    (output_base / method).mkdir(parents=True, exist_ok=True)

# Clase auxiliar para cargar imágenes en lotes
class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.image_files = [f for f in self.folder_path.glob('*.jpg') or self.folder_path.glob('*.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, str(img_path.name)

# Función para medir tiempo
def timer_func(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'{func.__name__}: {end - start:.4f} segundos')
        return result
    return wrapper

# Definir las transformaciones para PyTorch y TensorFlow/Keras
torch_transform = transforms.Compose([
    transforms.Resize((1920, 1920)),
])

tensorflow_transform = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resize(1920, 1920),
])

# Definir la función para hacer benchmarking
def benchmark(method_name, transform):
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Aplicar la transformación
            if method_name == 'PyTorch':
                img = torch_transform(img)
            elif method_name == 'TensorFlow':
                img = tf.expand_dims(load_img(os.path.join(source_folder, filename)), 0)
                img = tensorflow_transform(img)
                img = tf.squeeze(img, [0])
            
            # Guardar la imagen en una nueva carpeta
            img.save(os.path.join(source_folder, method_name, filename))
    
    end = time.time()
    print(f'{method_name}: {end - start} segundos')


# Definir la función para hacer benchmarking con NVIDIA NPP
def benchmark_npp():
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(source_folder, filename))  # Usamos OpenCV para leer la imagen
            
            # Aplicar transformación de redimensionamiento con NPP (NVIDIA Performance Primitives)
            dst = cv2.resize(img, (1920, 1920), interpolation=cv2.INTER_LINEAR)
            
            # Guardar la imagen en una nueva carpeta
            cv2.imwrite(os.path.join(source_folder, 'NVIDIA NPP', filename), dst)
    
    end = time.time()
    print('NVIDIA NPP: {} segundos'.format(end - start))



# Definir la función para hacer benchmarking con OpenCV con CUDA
def benchmark_opencv_cuda():
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(source_folder, filename))   # Usamos OpenCV para leer la imagen
            
            # Aplicar transformación de redimensionamiento con OpenCV (con soporte para CUDA)
            dst = cv2.cuda.resize(img, (1920, 1920))
            
            # Guardar la imagen en una nueva carpeta
            cv2.imwrite(os.path.join(source_folder, 'OpenCV with CUDA', filename), dst)
    
    end = time.time()
    print('OpenCV con CUDA: {} segundos'.format(end - start))

# Definir la función para hacer benchmarking con CuPy
def benchmark_cupy():
    import cupy as cp
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Convertir la imagen a un array de CuPy
            img_array = cp.array(img)
            
            # Aplicar transformación de redimensionamiento con CuPy
            img_resized = cp.resize(img_array, (1920, 1920))
            
            # Guardar la imagen en una nueva carpeta
            img_resized_pil = Image.fromarray(cp.asnumpy(img_resized))
            img_resized_pil.save(os.path.join(source_folder, 'CuPy', filename))
    
    end = time.time()
    print('CuPy: {} segundos'.format(end - start))

# Definir la función para hacer benchmarking con CuPy y TensorRT
def benchmark_cupy_tensorrt():
    import cupy as cp
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Convertir la imagen a un array de CuPy
            img_array = cp.array(img)
            
            # Aplicar transformación de redimensionamiento con Cu


def benchmark_tensorrt():
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Convertir la imagen a un array de CuPy
            img_array = cp.array(img)
            
            # Aplicar transformación de redimensionamiento con TensorRT (opcional, requiere configuración adicional)
            # Aquí se debe implementar el código específico para TensorRT
            
            # Guardar la imagen en una nueva carpeta
            img_resized_pil = Image.fromarray(cp.asnumpy(img_array))
            img_resized_pil.save(os.path.join(source_folder, 'TensorRT', filename))
    
    end = time.time()
    print('TensorRT: {} segundos'.format(end - start))

# Definir la función para hacer benchmarking con ArrayFire
def benchmark_arrayfire():
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Convertir la imagen a un array de ArrayFire
            img_array = af.to_array(img)
            
            # Aplicar transformación de redimensionamiento con ArrayFire
            img_resized = af.resize(img_array, (1920, 1920))
            
            # Guardar la imagen en una nueva carpeta
            img_resized_pil = Image.fromarray(af.to_numpy(img_resized))
            img_resized_pil.save(os.path.join(source_folder, 'ArrayFire', filename))
    
    end = time.time()
    print('ArrayFire: {} segundos'.format(end - start))


# Definir la función para hacer benchmarking con PyCUDA
def benchmark_pycuda():
    start = time.time()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_folder, filename))
            
            # Convertir la imagen a un array de PyCUDA
            img_array = cp.array(img)
            
            # Aplicar transformación de redimensionamiento con PyCUDA (opcional, requiere configuración adicional)
            # Aquí se debe implementar el código específico para PyCUDA
            
            # Guardar la imagen en una nueva carpeta
            img_resized_pil = Image.fromarray(cp.asnumpy(img_array))
            img_resized_pil.save(os.path.join(source_folder, 'PyCUDA', filename))
    
    end = time.time()
    print('PyCUDA: {} segundos'.format(end - start))

@timer_func
def benchmark_pytorch_batch():
    # Configurar el dispositivo CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear el dataset y dataloader
    dataset = ImageDataset(source_folder, transform=torch_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    for batch_images, batch_filenames in dataloader:
        batch_images = batch_images.to(device)
        resized_images = torch_transform(batch_images)
        
        # Guardar las imágenes procesadas
        for img, filename in zip(resized_images.cpu(), batch_filenames):
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(output_base / 'PyTorch_Batch' / filename)

@timer_func
def benchmark_tensorflow_batch():
    dataset = tf.data.Dataset.from_tensor_slices(list(source_folder.glob('*.jpg')))
    dataset = dataset.map(lambda x: tf.io.read_file(x))
    dataset = dataset.map(lambda x: tf.io.decode_jpeg(x, channels=3))
    dataset = dataset.batch(BATCH_SIZE)

    for batch in dataset:
        resized = tensorflow_transform(batch)
        for i, img in enumerate(resized):
            tf.keras.preprocessing.image.save_img(
                output_base / 'TensorFlow_Batch' / f'img_{i}.jpg', 
                img.numpy()
            )

@timer_func
def benchmark_opencv_cuda_batch():
    # Corregir el método anterior que usaba cuda_FHDML
    images = []
    filenames = []
    
    # Recopilar imágenes para el procesamiento por lotes
    for file in source_folder.glob('*.jpg'):
        img = cv2.imread(str(file))
        if img is not None:
            images.append(img)
            filenames.append(file.name)
            
            if len(images) == BATCH_SIZE:
                # Crear stream CUDA
                stream = cv2.cuda_Stream()
                
                # Subir lote a GPU
                gpu_images = [cv2.cuda.GpuMat(img) for img in images]
                
                # Redimensionar en paralelo
                resized = [cv2.cuda.resize(img, TARGET_SIZE) for img in gpu_images]
                
                # Descargar y guardar
                for img, filename in zip(resized, filenames):
                    result = img.download()
                    cv2.imwrite(str(output_base / 'OpenCV_CUDA_Batch' / filename), result)
                
                images = []
                filenames = []
    
    # Procesar las imágenes restantes
    if images:
        stream = cv2.cuda_Stream()
        gpu_images = [cv2.cuda.GpuMat(img) for img in images]
        resized = [cv2.cuda.resize(img, TARGET_SIZE) for img in gpu_images]
        for img, filename in zip(resized, filenames):
            result = img.download()
            cv2.imwrite(str(output_base / 'OpenCV_CUDA_Batch' / filename), result)

@timer_func
def benchmark_cupy_batch():
    images = []
    filenames = []
    
    for file in source_folder.glob('*.jpg'):
        img = cp.array(Image.open(file))
        images.append(img)
        filenames.append(file.name)
        
        if len(images) == BATCH_SIZE:
            # Procesar lote
            batch = cp.stack(images)
            resized = cp.ndarray((len(batch), *TARGET_SIZE, 3))
            
            for i, img in enumerate(batch):
                resized[i] = cp.resize(img, TARGET_SIZE)
            
            # Guardar resultados
            for img, filename in zip(resized, filenames):
                img_pil = Image.fromarray(cp.asnumpy(img).astype(np.uint8))
                img_pil.save(output_base / 'CuPy_Batch' / filename)
            
            images = []
            filenames = []
    
    # Procesar imágenes restantes
    if images:
        batch = cp.stack(images)
        resized = cp.ndarray((len(batch), *TARGET_SIZE, 3))
        for i, img in enumerate(batch):
            resized[i] = cp.resize(img, TARGET_SIZE)
        
        for img, filename in zip(resized, filenames):
            img_pil = Image.fromarray(cp.asnumpy(img).astype(np.uint8))
            img_pil.save(output_base / 'CuPy_Batch' / filename)

@timer_func
def benchmark_arrayfire_batch():
    images = []
    filenames = []
    
    for file in source_folder.glob('*.jpg'):
        img = af.to_array(np.array(Image.open(file)))
        images.append(img)
        filenames.append(file.name)
        
        if len(images) == BATCH_SIZE:
            # Procesar lote
            batch = af.join(0, *images)  # Unir imágenes en un solo tensor
            resized = af.resize(batch, *TARGET_SIZE)
            
            # Guardar resultados
            for i, filename in enumerate(filenames):
                img = af.to_array(resized[i])
                img_pil = Image.fromarray(img.to_numpy().astype(np.uint8))
                img_pil.save(output_base / 'ArrayFire_Batch' / filename)
            
            images = []
            filenames = []
    
    # Procesar imágenes restantes
    if images:
        batch = af.join(0, *images)
        resized = af.resize(batch, *TARGET_SIZE)
        
        for i, filename in enumerate(filenames):
            img = af.to_array(resized[i])
            img_pil = Image.fromarray(img.to_numpy().astype(np.uint8))
            img_pil.save(output_base / 'ArrayFire_Batch' / filename)

@timer_func
def benchmark_pycuda_batch():
    # Definir kernel CUDA para redimensionamiento
    kernel_code = """
    __global__ void resize_kernel(float *input, float *output, int in_w, int in_h, int out_w, int out_h) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < out_w * out_h) {
            int out_x = idx % out_w;
            int out_y = idx / out_w;
            
            float scale_x = float(in_w) / out_w;
            float scale_y = float(in_h) / out_h;
            
            int in_x = int(out_x * scale_x);
            int in_y = int(out_y * scale_y);
            
            output[idx] = input[in_y * in_w + in_x];
        }
    }
    """
    
    # Compilar el kernel
    mod = cuda.module_from_source(kernel_code)
    resize_kernel = mod.get_function("resize_kernel")
    
    images = []
    filenames = []
    
    for file in source_folder.glob('*.jpg'):
        img = np.array(Image.open(file)).astype(np.float32)
        images.append(img)
        filenames.append(file.name)
        
        if len(images) == BATCH_SIZE:
            # Procesar lote
            batch = np.stack(images)
            input_gpu = gpuarray.to_gpu(batch)
            output = np.empty((len(batch), *TARGET_SIZE, 3), dtype=np.float32)
            output_gpu = gpuarray.empty_like(output)
            
            # Ejecutar kernel para cada imagen en el lote
            block_size = (256, 1, 1)
            grid_size = ((TARGET_SIZE[0] * TARGET_SIZE[1] + block_size[0] - 1) // block_size[0], 1)
            
            for i in range(len(batch)):
                resize_kernel(
                    input_gpu[i].gpudata,
                    output_gpu[i].gpudata,
                    np.int32(batch.shape[2]),
                    np.int32(batch.shape[1]),
                    np.int32(TARGET_SIZE[0]),
                    np.int32(TARGET_SIZE[1]),
                    block=block_size,
                    grid=grid_size
                )
            
            # Guardar resultados
            output = output_gpu.get()
            for img, filename in zip(output, filenames):
                img_pil = Image.fromarray(img.astype(np.uint8))
                img_pil.save(output_base / 'PyCUDA_Batch' / filename)
            
            images = []
            filenames = []
    
    # Procesar imágenes restantes
    if images:
        batch = np.stack(images)
        input_gpu = gpuarray.to_gpu(batch)
        output = np.empty((len(batch), *TARGET_SIZE, 3), dtype=np.float32)
        output_gpu = gpuarray.empty_like(output)
        
        for i in range(len(batch)):
            resize_kernel(
                input_gpu[i].gpudata,
                output_gpu[i].gpudata,
                np.int32(batch.shape[2]),
                np.int32(batch.shape[1]),
                np.int32(TARGET_SIZE[0]),
                np.int32(TARGET_SIZE[1]),
                block=block_size,
                grid=grid_size
            )
        
        output = output_gpu.get()
        for img, filename in zip(output, filenames):
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_pil.save(output_base / 'PyCUDA_Batch' / filename)

if __name__ == '__main__':
    print("Ejecutando benchmarks...")
    
    # Ejecutar versiones single
    benchmark('PyTorch', torch_transform)
    benchmark('TensorFlow', tensorflow_transform)
    benchmark_opencv_cuda()
    benchmark_cupy()
    benchmark_arrayfire()
    benchmark_pycuda()
    
    print("\nEjecutando versiones batch...")
    
    # Ejecutar versiones batch
    benchmark_pytorch_batch()
    benchmark_tensorflow_batch()
    benchmark_opencv_cuda_batch()
    benchmark_cupy_batch()
    benchmark_arrayfire_batch()
    benchmark_pycuda_batch()
    
    print("\n¡Benchmarks completados!")