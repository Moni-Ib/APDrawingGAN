import cv2
import os

input_folder = "dataset/cats_photos"
output_folder = "dataset/cats_sketches"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # Hacemos resize 
        img = cv2.resize(img, (256, 256))

        # Convertir a gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Suavizar
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Bordes (sketch)
        edges = cv2.Canny(blur, 50, 150)

        # Invertir colores
        sketch = cv2.bitwise_not(edges)

        # Guardar
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, sketch)

print("Sketches generados")