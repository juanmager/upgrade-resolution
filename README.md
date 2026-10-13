# Upgrade Resolution

Mejora la calidad y la resolución de imágenes (fotos, capturas de pantalla, JPG,
PNG, WebP, BMP, TIFF) con IA. Usa redes **Real-ESRGAN**, que reconstruyen el
detalle y eliminan el ruido, el desenfoque y los artefactos de compresión JPEG.

## Instalación (Windows con GPU AMD Radeon RX 7000)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --index-url https://repo.amd.com/rocm/whl/gfx110X-dgpu/ torch torchvision
.venv\Scripts\python -m pip install -e ".[dev]"
```

PyTorch con ROCm expone la GPU AMD como dispositivo `cuda`, y el programa la
detecta automáticamente. Con GPU NVIDIA, instala el `torch` de
<https://pytorch.org>; sin GPU, el `torch` normal funciona en CPU, pero mucho más
lento.

Los pesos de cada modelo se descargan la primera vez que se usan, se verifican
con SHA-256 y se guardan en `~/.cache/upgrade_resolution`. Se puede cambiar esa
carpeta con la variable de entorno `UPGRADE_RESOLUTION_CACHE`.

## Uso

```powershell
upgrade-resolution foto.jpg                      # x4 con el modelo "photo"
upgrade-resolution captura.png -m anime          # capturas, texto, ilustraciones
upgrade-resolution foto.jpg -s 2                 # x2
upgrade-resolution foto.jpg -s 1                 # mismo tamaño, solo mejora la calidad
upgrade-resolution C:\fotos -o C:\mejoradas -f webp   # carpeta completa
```

Los resultados se guardan como `<nombre>_x<escala>.<formato>` en una carpeta
`upscaled` junto a cada imagen, salvo que se indique `-o`. Si un resultado ya
existe, se omite; usa `--overwrite` para reemplazarlo. Ejecuta
`upgrade-resolution --help` para ver todas las opciones.

| Modelo    | Ideal para                                             | Velocidad |
|-----------|--------------------------------------------------------|-----------|
| `photo`   | Fotos reales; máxima calidad (opción por defecto)       | Media     |
| `anime`   | Capturas de pantalla, texto, dibujos, colores planos   | Rápida    |
| `general` | Uso general cuando prima la velocidad                  | Rápida    |

La primera vez que aparece cada tamaño de bloque, ROCm compila sus kernels y
tarda alrededor de un segundo más. Si aparece un error de memoria de GPU,
reduce el tamaño de bloque, por ejemplo con `--tile 256`.

Cada bloque se compara con su entrada. Si la red devuelve un resultado que no
puede ser una versión de la imagen (negro, o con valores inválidos), se repite
en precisión completa y, si sigue mal, se informa el error sin guardar nada.

### GPU AMD en Windows

Al ver por primera vez cada forma de convolución, MIOpen (la biblioteca de
convoluciones de ROCm) mide todos sus kernels candidatos. En Windows, algunos
(GEMM, directos) tardan más de un segundo o provocan un fallo de página en la
GPU: el controlador se reinicia (aviso de AMD de tiempo de espera agotado,
evento `LiveKernelEvent 0x141` con `PF:1`) y el bloque sale negro. Por eso el
programa habilita solo los kernels Winograd, que cubren todas las convoluciones
3x3 de estos modelos y tardan menos de un milisegundo por capa. Se puede
cambiar con las variables `MIOPEN_*` de `MIOPEN_SETTINGS` en `upscaler.py`: si
ya están definidas en el entorno, el programa respeta su valor.

## Diseño

El código sigue los principios del curso MIT 6.031 (*Software Construction*):
que sea seguro ante errores, fácil de entender y esté preparado para cambios.

- **Especificaciones** en cada función pública: precondiciones (`Args`),
  postcondiciones (`Returns`) y excepciones (`Raises`).
- **Tipos abstractos de datos** con función de abstracción, invariante de
  representación y argumento de seguridad frente a la exposición de la
  representación (`RasterImage`, `Upscaler`).
- **Inmutabilidad:** `RasterImage` congela sus arrays, y `ModelSpec` y el
  registro `MODELS` son de solo lectura.
- **Fallar rápido:** los argumentos se validan al entrar, y los pesos se
  verifican antes de guardarlos en la caché.
- **Bajo acoplamiento:** el procesamiento por bloques es una función pura e
  independiente de la red. La red se inyecta mediante un `Protocol`, y la CLI
  recibe una fábrica, por lo que todo se prueba sin descargar modelos.
- **Pruebas por particiones:** cada archivo de pruebas documenta su estrategia
  de partición del espacio de entradas.

| Módulo        | Responsabilidad                                            |
|---------------|------------------------------------------------------------|
| `models.py`   | Registro de modelos, descarga y verificación de pesos      |
| `image_io.py` | Lectura y escritura de imágenes (EXIF, transparencia)      |
| `tiling.py`   | Procesamiento por bloques sin costuras                     |
| `upscaler.py` | Ejecución de la red en GPU o CPU y reescalado final        |
| `cli.py`      | Interfaz de línea de comandos                              |

## Desarrollo

```powershell
.venv\Scripts\python -m pytest     # pruebas
.venv\Scripts\ruff check src tests # estilo (PEP 8, docstrings, imports)
.venv\Scripts\ruff format src tests
.venv\Scripts\mypy                 # tipos en modo estricto
```
