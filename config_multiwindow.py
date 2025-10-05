"""
Configuración para el NASA Multi-Window Image Explorer
"""

# Configuración de ventanas
WINDOW_CONFIG = {
    'default_width': 1400,
    'default_height': 900,
    'max_windows': 20,  # Límite de ventanas simultáneas
    'auto_arrange': True,  # Organizar ventanas automáticamente
}

# Configuración de imágenes
IMAGE_CONFIG = {
    'supported_formats': ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp', '*.fits'],
    'max_image_size_mb': 100,  # Tamaño máximo de imagen
    'preload_filters': True,   # Precargar filtros automáticamente
}

# Configuración de interfaz
UI_CONFIG = {
    'theme': 'default',  # 'default', 'dark', 'light'
    'language': 'es',    # 'es', 'en'
    'show_previews': True,
}

# Configuración de rendimiento
PERFORMANCE_CONFIG = {
    'use_tiling': True,      # Usar sistema de tiles para imágenes grandes
    'tile_size': 512,        # Tamaño de tiles
    'cache_size': 10,        # Número de imágenes en caché
}