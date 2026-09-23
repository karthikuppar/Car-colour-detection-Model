import cv2
import numpy as np
from ultralytics import YOLO
from backend.src.classification.color_detector import CarColorDetector

class TrafficAnnotator:
    def __init__(self, yolo_weights='yolov8s.pt'):
        self.detector = YOLO(yolo_weights)
        self.color_classifier = CarColorDetector()

        # Task colors: RED for blue cars, BLUE for other cars
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_GREEN = (0, 255, 0)

    def process_and_annotate(self, image_path, conf=0.25, iou=0.5):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f'Could not load {image_path}')

        h, w, _ = img.shape

        results = self.detector.predict(
            source=img,
            conf=conf,
            iou=iou,
            agnostic_nms=True,
            classes=[0, 2, 5, 7],
            imgsz=1280,
            verbose=False
        )[0]

        counts = {
            'total_cars': 0,
            'blue_cars': 0,
            'other_cars': 0,
            'people_count': 0
        }

        boxes = results.boxes
        if len(boxes) == 0:
            return img, counts

        for box in boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw = x2 - x1
            bh = y2 - y1

            # Case A: Pedestrian
            # Upright human posture check: height must exceed width to prevent headrests/mirrors from being flagged
            if cls_id == 0:
                if bh < int(bw * 1.15) or bh < 24:
                    continue  # Ignore squarish/wide reflections inside cars
                counts['people_count'] += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), self.COLOR_GREEN, 2)
                cv2.putText(img, 'Person', (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_GREEN, 1, cv2.LINE_AA)

            # Case B: Vehicles (car, bus, truck)
            elif cls_id in [2, 5, 7]:
                counts['total_cars'] += 1
                crop = img[y1:y2, x1:x2]

                is_blue, blue_conf, color_label = self.color_classifier.is_blue_car(crop)

                if is_blue:
                    counts['blue_cars'] += 1
                    box_color = self.COLOR_RED
                    label_text = f'Blue Car ({blue_conf:.2f})'
                else:
                    counts['other_cars'] += 1
                    box_color = self.COLOR_BLUE
                    label_text = 'Other Car'

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

                # Label tag
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (x1, max(y1 - 18, 0)), (x1 + tw + 4, max(y1, 18)), box_color, -1)
                cv2.putText(img, label_text, (x1 + 2, max(y1 - 4, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        self._draw_dashboard(img, counts)
        return img, counts

    def _draw_dashboard(self, img, counts):
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (430, 130), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        cv2.rectangle(img, (10, 10), (430, 130), (255, 255, 255), 1)
        cv2.putText(img, 'TRAFFIC SIGNAL ANALYTICS', (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        stats = [
            (f'Total Vehicles: {counts["total_cars"]}', (240, 240, 240)),
            (f'Blue Cars [RED BOX]: {counts["blue_cars"]}', (50, 50, 255)),
            (f'Other Cars [BLUE BOX]: {counts["other_cars"]}', (255, 150, 50)),
            (f'Pedestrians: {counts["people_count"]}', (100, 255, 100))
        ]

        y_offset = 56
        for text, color in stats:
            cv2.putText(img, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            y_offset += 20

if __name__ == '__main__':
    annotator = TrafficAnnotator()
    output_img, summary = annotator.process_and_annotate('dataset/image.png')
    out_file = 'dataset/final_annotated_image.jpg'
    cv2.imwrite(out_file, output_img)
    print('='*50)
    print('FINAL ANNOTATION FINISHED!')
    print('Counts Summary:', summary)
    print(f'Annotated file written to: {out_file}')
    print('='*50)
