from ultralytics import YOLO

#Model Yükle 

model= YOLO('yolov8n.yaml') #Yaml'dan yeni bir model oluştur
model=YOLO('yolov8n.pt') # Önceden eğitilmiş bir model yükleyin
model= YOLO('yolov8n.yaml').load('yolov8n.pt') #YAML'den derleme ve ağırlıkları aktarma


#Modeli Eğit

result= model.train(data='coco8.yaml', epochs=25) 