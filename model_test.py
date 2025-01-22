from ultralytics import YOLO

model = YOLO('./yolo_MyGO.pt')  

input_path = './test.jpg' # image, video
model.predict(input_path, conf=0.4, save=True, device=0)

