import cv2
import os

def crop_vehicles(image_path, detections, output_dir='dataset/crops'):
    '''
    Crops detected vehicles from an image and saves them to disk for inspection.
    
    Args:
        image_path: Path to the input image
        detections: List of bounding boxes [[x1, y1, x2, y2, conf, cls_id], ...]
        output_dir: Directory where cropped cars will be saved
    Returns:
        List of paths to saved crop images
    '''
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f'Could not load image: {image_path}')
        
    h, w, _ = img.shape
    crop_paths = []
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(val) for val in det[:4]]
        
        # Clamp coordinates to image boundaries to prevent crashes
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Ensure the crop has valid dimensions (at least 20x20 pixels)
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
            
        crop = img[y1:y2, x1:x2]
        crop_filename = os.path.join(output_dir, f'crop_{idx}.jpg')
        cv2.imwrite(crop_filename, crop)
        crop_paths.append(crop_filename)
        
    return crop_paths

if __name__ == '__main__':
    from ultralytics import YOLO
    
    test_img = 'dataset/clean_detection.jpg'
    model = YOLO('yolov8s.pt')
    results = model.predict(r'dataset/image copy.png', conf=0.25, iou=0.5, agnostic_nms=True, classes=[2, 5, 7], verbose=False)[0]
    
    boxes = [b.xyxy[0].tolist() for b in results.boxes]
    paths = crop_vehicles(r'dataset/image copy.png', boxes)
    print(f'[SUCCESS] Extracted and saved {len(paths)} vehicle crops into dataset/crops/')
