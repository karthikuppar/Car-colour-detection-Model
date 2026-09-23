import torch
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self, model_name='yolov8n.pt'):
        # Check if CUDA (RTX 4050) is available, otherwise fall back to CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[INFO] Loading YOLO detector ({model_name}) on device: {self.device}')
        self.model = YOLO(model_name)
        
        # Target COCO class indices: 0 is person, 2 is car
        # 3 is motorcycle, 5 is bus, 7 is truck (can be added if desired)
        self.CAR_CLASS_ID = 2
        self.PERSON_CLASS_ID = 0

    def detect(self, image_path, conf_threshold=0.35):
        '''
        Runs detection on an image.
        Returns:
            car_boxes: list of [x1, y1, x2, y2, confidence]
            person_boxes: list of [x1, y1, x2, y2, confidence]
            raw_result: Ultralytics result object
        '''
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes
        
        car_boxes = []
        person_boxes = []
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            entry = [round(c, 2) for c in coords] + [round(conf, 2)]
            
            if cls_id == self.CAR_CLASS_ID:
                car_boxes.append(entry)
            elif cls_id == self.PERSON_CLASS_ID:
                person_boxes.append(entry)
                
        return car_boxes, person_boxes, result

if __name__ == '__main__':
    detector = TrafficDetector()
    test_img = 'dataset/traffic_test.jpg'
    cars, people, _ = detector.detect(test_img)
    
    print('-----------------------------------------')
    print(f'Total Cars Detected: {len(cars)}')
    print(f'Total People Detected: {len(people)}')
    print('Sample Car Bounding Box (x1, y1, x2, y2, conf):', cars[0] if cars else 'None')
    print('-----------------------------------------')
