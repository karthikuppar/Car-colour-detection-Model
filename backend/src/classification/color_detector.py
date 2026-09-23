import cv2
import numpy as np

class CarColorDetector:
    '''
    Robust Vehicle Color Detector using Multi-Space Chromaticity Analysis (CIE-LAB + HSV + RGB)
    with Multi-Region Sub-Quadrant Spatial Scanning.
    
    Accurately classifies vehicles as 'blue' or 'other' across lighting conditions, shadows,
    and partial occlusions while rejecting clothing (e.g. blue denim jackets) and warm/neutral reflections.
    '''
    def __init__(self):
        self.NEUTRAL_LAB_B = 128.0
        self.last_debug_stats = {}

    def is_blue_car(self, car_crop):
        '''
        Main classification interface.
        Returns:
            is_blue (bool): True if the car is classified as blue paint.
            confidence (float): Technical confidence score between 0.0 and 1.0.
            color_label (str): 'blue' or 'other'.
        '''
        if car_crop is None or car_crop.size == 0:
            self.last_debug_stats = {'reason': 'empty_crop'}
            return False, 0.0, 'other'

        h, w, _ = car_crop.shape
        if h < 12 or w < 12:
            self.last_debug_stats = {'reason': 'crop_too_small'}
            return False, 0.0, 'other'

        # Standardized 64x64 resolution for noise reduction and spatial region extraction
        body_res = cv2.resize(car_crop, (64, 64), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(body_res, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(body_res, cv2.COLOR_BGR2LAB)

        b_bgr = body_res[:, :, 0].astype(np.float32)
        g_bgr = body_res[:, :, 1].astype(np.float32)
        r_bgr = body_res[:, :, 2].astype(np.float32)

        h_chan = hsv[:, :, 0].astype(np.float32)  # OpenCV range: 0..180
        s_chan = hsv[:, :, 1].astype(np.float32)  # OpenCV range: 0..255
        v_chan = hsv[:, :, 2].astype(np.float32)  # OpenCV range: 0..255

        l_lab = lab[:, :, 0].astype(np.float32)    # Lightness: 0..255
        a_lab = lab[:, :, 1].astype(np.float32)    # Green-Red axis (128 neutral)
        b_lab = lab[:, :, 2].astype(np.float32)    # Blue-Yellow axis (128 neutral, <128 is blue)

        # Filter extreme highlights (glints/sky reflection on glass) and deep shadows
        valid_mask = (v_chan >= 18) & (v_chan <= 245) & (l_lab >= 15) & (l_lab <= 245)
        if np.sum(valid_mask) < 40:
            valid_mask = np.ones((64, 64), dtype=bool)

        # Chromaticity metrics
        chroma = np.sqrt((a_lab - 128.0)**2 + (b_lab - 128.0)**2)
        max_rg = np.maximum(r_bgr, g_bgr)
        blue_dom = (b_bgr - max_rg) / (b_bgr + max_rg + 1.0)

        # Glass Reflection & Neutral Rejection Filter
        glass_reflection_mask = (l_lab > 165) & (s_chan < 45) & (b_lab > 112.0)
        neutral_pixel_mask = valid_mask & (
            (chroma < 7.0) |
            ((s_chan < 25) & (b_lab >= 123.5)) |
            (blue_dom < 0.02) |
            glass_reflection_mask
        )

        # Blue Paint Pixel Identification Rules
        blue_pixel_mask = valid_mask & (
            (h_chan >= 92) & (h_chan <= 138) &
            (b_lab <= 122.0) &
            (b_bgr >= r_bgr + 4) &
            (b_bgr >= g_bgr) &
            (s_chan >= 30) &
            (blue_dom >= 0.03)
        ) & (~neutral_pixel_mask)

        n_valid = int(np.sum(valid_mask))
        n_blue = int(np.sum(blue_pixel_mask))
        n_neutral = int(np.sum(neutral_pixel_mask))

        blue_pixel_ratio = n_blue / n_valid if n_valid > 0 else 0.0
        neutral_pixel_ratio = n_neutral / n_valid if n_valid > 0 else 0.0

        body_b_lab_mean = float(np.mean(b_lab[valid_mask]))
        blue_b_lab_mean = float(np.mean(b_lab[blue_pixel_mask])) if n_blue > 0 else 128.0
        blue_sat_mean = float(np.mean(s_chan[blue_pixel_mask])) if n_blue > 0 else 0.0

        # Multi-Region Spatial Sub-Quadrant Scan
        # Evaluates Top Half, Left, Right, Top-Left 60%, Top-Right 60%, Center 60%
        regions = [
            blue_pixel_mask[0:32, :],         # Top Half
            blue_pixel_mask[:, 0:32],         # Left Half
            blue_pixel_mask[:, 32:64],        # Right Half
            blue_pixel_mask[0:40, 0:40],      # Top-Left 60%
            blue_pixel_mask[0:40, 24:64],     # Top-Right 60%
            blue_pixel_mask[12:52, 12:52]     # Center 60%
        ]
        max_region_blue_ratio = max(float(np.mean(r)) for r in regions)

        # Technical Confidence Score Calculation
        score_global_ratio = min(1.0, blue_pixel_ratio / 0.35)
        score_region_ratio = min(1.0, max_region_blue_ratio / 0.28)
        score_blab = min(1.0, max(0.0, (125.0 - blue_b_lab_mean) / 10.0)) if n_blue > 0 else 0.0
        score_sat = min(1.0, blue_sat_mean / 100.0) if n_blue > 0 else 0.0

        blue_score = 0.35 * score_region_ratio + 0.30 * score_global_ratio + 0.20 * score_blab + 0.15 * score_sat
        blue_score = min(1.0, max(0.0, blue_score))

        # Binary Classification Logic
        # Genuine blue car paint requires either strong whole-body dominance OR a vivid sub-region paint signal
        is_blue = (
            (blue_pixel_ratio >= 0.22 and body_b_lab_mean <= 124.5 and blue_b_lab_mean <= 118.0) or
            (max_region_blue_ratio >= 0.22 and blue_b_lab_mean <= 114.0 and blue_sat_mean >= 90.0)
        )

        # Neutral & Muted Clothing Safety Override:
        if neutral_pixel_ratio >= 0.65 and blue_pixel_ratio < 0.20 and body_b_lab_mean > 124.0 and blue_b_lab_mean > 112.0:
            is_blue = False

        confidence = round(blue_score if is_blue else (1.0 - blue_score), 2)
        color_label = 'blue' if is_blue else 'other'

        self.last_debug_stats = {
            'blue_pixel_ratio': round(blue_pixel_ratio, 3),
            'max_region_blue_ratio': round(max_region_blue_ratio, 3),
            'body_b_lab_mean': round(body_b_lab_mean, 1),
            'blue_b_lab_mean': round(blue_b_lab_mean, 1),
            'blue_sat_mean': round(blue_sat_mean, 1),
            'blue_score': round(blue_score, 3)
        }

        return is_blue, confidence, color_label

