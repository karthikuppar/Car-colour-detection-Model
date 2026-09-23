import cv2
import numpy as np

class CarColorDetector:
    def __init__(self):
        # In OpenCV HSV, blue hue strictly spans 98 to 132
        self.HUE_MIN = 98
        self.HUE_MAX = 132

    def is_blue_car(self, car_crop):
        '''
        Reliably identifies blue vehicles across lighting conditions:
        - Light sky blue
        - Royal/electric blue
        - Deep navy / midnight blue
        Rejects silver, white, black, gold, and red.
        '''
        if car_crop is None or car_crop.size == 0:
            return False, 0.0, 'other'

        h, w, _ = car_crop.shape
        if h < 16 or w < 16:
            return False, 0.0, 'other'

        # Focus on the rear/trunk body sheet metal:
        # Avoid the upper window (top 35%) and road tarmac shadow (bottom 20%)
        y1 = int(h * 0.35)
        y2 = int(h * 0.80)
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)

        body = car_crop[y1:y2, x1:x2]
        if body.size == 0:
            body = car_crop

        body_small = cv2.resize(body, (50, 50))
        pixels = body_small.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        counts = np.bincount(labels.flatten())
        sorted_indices = np.argsort(counts)[::-1]
        centers_bgr = np.uint8(centers)

        for idx in sorted_indices:
            b, g, r = [float(v) for v in centers_bgr[idx]]
            proportion = counts[idx] / len(labels)

            cluster_hsv = cv2.cvtColor(centers_bgr[idx].reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0][0]
            hue = int(cluster_hsv[0])
            sat = int(cluster_hsv[1])
            val = int(cluster_hsv[2])

            # 1. Hue condition: must fall in the blue spectrum
            has_blue_hue = (self.HUE_MIN <= hue <= self.HUE_MAX)

            # 2. Saturation condition: must be colorful paint, not neutral gray/silver/white
            has_saturation = (sat >= 35)

            # 3. Minimum brightness to avoid pure black rubber
            has_visibility = (val >= 20)

            # 4. Proportional Channel Dominance: Blue must clearly lead Red & Green
            # This works for dark navy (e.g. B=30, G=22, R=21) as well as bright blues
            leads_red = (b >= r * 1.18) or (b >= r + 7)
            leads_green = (b >= g * 1.08) or (b >= g + 4)
            is_dominant_channel = leads_red and leads_green

            # 5. Significance: Cluster must represent at least 22% of sampled body
            is_significant = (proportion >= 0.22)

            if has_blue_hue and has_saturation and has_visibility and is_dominant_channel and is_significant:
                return True, round(proportion, 2), 'blue'

        return False, 0.0, 'other'
