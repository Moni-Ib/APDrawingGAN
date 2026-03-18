import cv2
import os
import numpy as np

input_folder = "dataset/cats_photos"
output_folder = "dataset/cats/test"

os.makedirs(output_folder, exist_ok=True)

count = 0
max_test = 100 

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        
        if count >= max_test:
            break
        
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # Resize
        img = cv2.resize(img, (256, 256))

        # GENERAR SKETCH
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        sketch = cv2.bitwise_not(edges)

        # Convertir sketch a 3 canales
        sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        # UNIR FOTO + SKETCH
        AB = np.concatenate((img, sketch), axis=1)

        # Guardar
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, AB)

        count += 1

print("Test dataset creado")