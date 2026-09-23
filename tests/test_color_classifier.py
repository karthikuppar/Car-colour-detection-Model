import os
import cv2
from backend.src.classification.color_detector import CarColorDetector

def run_color_classification_test(crops_dir='dataset/crops_image'):
    detector = CarColorDetector()

    if not os.path.exists(crops_dir):
        print(f"Error: Directory '{crops_dir}' not found.")
        return

    crop_files = sorted(
        [f for f in os.listdir(crops_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x and x.split('_')[1].split('.')[0].isdigit() else 999
    )

    if not crop_files:
        print(f"No image crops found in '{crops_dir}'.")
        return

    print("==================================================")
    print("CAR COLOUR CLASSIFICATION TEST")
    print("==================================================")

    blue_count = 0
    other_count = 0

    for fname in crop_files:
        img_path = os.path.join(crops_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Image: {fname}\nCould not load image file.\n")
            continue

        is_blue, confidence, label = detector.is_blue_car(img)
        stats = detector.last_debug_stats

        if is_blue:
            blue_count += 1
            pred_text = "BLUE"
        else:
            other_count += 1
            pred_text = "OTHER"

        print(f"Image: {fname}")
        print(f"Prediction: {pred_text}")
        print(f"Confidence: {confidence:.2f}")

        if stats and 'blue_score' in stats:
            print(f"  Blue score: {stats['blue_score']:.2f}")
            print(f"  Blue pixel ratio: {stats['blue_pixel_ratio']:.3f}")
            print(f"  Max sub-region blue ratio: {stats['max_region_blue_ratio']:.3f}")
            print(f"  Body mean Lab b*: {stats['body_b_lab_mean']:.1f}")
            print(f"  Blue paint mean Lab b*: {stats['blue_b_lab_mean']:.1f}")
        print("-" * 50)

    print("==================================================")
    print(f"SUMMARY: Total Crops: {len(crop_files)} | Blue Cars: {blue_count} | Other Cars: {other_count}")
    print("==================================================")

if __name__ == '__main__':
    run_color_classification_test()
