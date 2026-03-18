import cv2
import os

input_folder = "dataset/cats_photos"
output_folder = "dataset/cats_sketches"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invertir
        inv = 255 - gray

        # Blur fuerte
        blur = cv2.GaussianBlur(inv, (21, 21), 0)

        # Dodge blend (efecto lápiz)
        sketch = cv2.divide(gray, 255 - blur, scale=256)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, sketch)

print("Sketches generados")