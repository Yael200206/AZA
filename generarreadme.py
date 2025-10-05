import os

# Carpeta con las imágenes
img_folder = "img"

# Lista todos los archivos en la carpeta
images = [f for f in os.listdir(img_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]

# Crear el contenido del README
readme_content = "# 🖼️ Galería de imágenes\n\n"

for img in images:
    readme_content += f"![{img}](img/{img})\n\n"

# Guardar el README
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("✅ README generado con las imágenes.")
