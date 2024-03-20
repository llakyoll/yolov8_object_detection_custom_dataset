#https://docs.ultralytics.com/modes/predict/

from ultralytics import YOLO
from PIL import Image
import cv2

model=YOLO("best.pt")
#Tahminleri göster. Tüm YOLO tahmin argümanlarını kabul eder
results= model.predict(source= "C:/Python/first_try\Predict/video2.mp4" , save=True) 

# PIL'den

# im1= Image.open("")
# results=model.predict(source=im1, save=True)
