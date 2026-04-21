import cv2
import numpy as np
import logging

class KtpDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_and_crop(self, image: np.ndarray, save_prefix: str = None) -> np.ndarray:
        """
        Detects a KTP card using a multi-stage hybrid detection pipeline.
        Returns a cropped, perspective-corrected version or the original if fails.
        """
        if image is None:
            return None

        orig = image.copy()
        try:
            # --- PREPARATION ---
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            l, a, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            img_enhanced = cv2.merge((cl, a, b_chan))
            img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_Lab2BGR)

            hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)

            # --- STAGE 1: COLOR MASKING ---
            lower_blue = np.array([85, 12, 20]) 
            upper_blue = np.array([145, 255, 255])
            hsv_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            ycrcb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2YCrCb)
            lower_skin = np.array([0, 133, 77], dtype="uint8")
            upper_skin = np.array([255, 173, 127], dtype="uint8")
            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
            not_skin_mask = cv2.bitwise_not(skin_mask)
            
            mask = cv2.bitwise_and(hsv_mask, not_skin_mask)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
            mask = cv2.dilate(mask, rect_kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horiz_kernel)
            
            screen_cnt = self._find_card_contour(mask, image.shape)
            working_mask = mask

            # --- STAGE 2: REGION-BASED "BLOB" DETECTION ---
            if screen_cnt is None:
                b_channel = img_enhanced[:, :, 0]
                blur = cv2.bilateralFilter(b_channel, 9, 75, 75)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                big_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, big_kernel, iterations=2)
                screen_cnt = self._find_card_contour(morphed, image.shape)
                if screen_cnt is None:
                    working_mask = morphed

            # --- STAGE 3: SPECTRAL UNION ---
            if screen_cnt is None:
                red_c = img_enhanced[:, :, 2]
                blue_c = img_enhanced[:, :, 0]
                clahe_edges = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                red_en = clahe_edges.apply(red_c)
                blue_en = clahe_edges.apply(blue_c)
                red_blur = cv2.bilateralFilter(red_en, 11, 80, 80)
                blue_blur = cv2.bilateralFilter(blue_en, 11, 80, 80)
                red_thresh = cv2.adaptiveThreshold(red_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
                blue_thresh = cv2.adaptiveThreshold(blue_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
                combined_mask = cv2.max(red_thresh, blue_thresh)
                screen_cnt = self._find_card_contour(combined_mask, image.shape)
                if screen_cnt is None:
                    working_mask = combined_mask

            # --- STAGE 4: MIN AREA RECT FALLBACK ---
            if screen_cnt is None:
                all_contours, _ = cv2.findContours(working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if all_contours:
                    largest_c = max(all_contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_c) > (image.shape[0] * image.shape[1] * 0.05):
                        rect = cv2.minAreaRect(largest_c)
                        box = cv2.boxPoints(rect)
                        screen_cnt = np.int64(box).reshape(4, 1, 2)

            # --- FINAL PROCESSING ---
            if screen_cnt is not None:
                warped = self._four_point_transform(image, gray, screen_cnt)
                return warped
            else:
                self.logger.warning("No KTP card detected using multi-stage logic.")
                return orig

        except Exception as e:
            self.logger.error(f"Detection pipeline error: {str(e)}")
            return orig

    def _find_card_contour(self, mask, img_shape):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            area = cv2.contourArea(c)
            if area < (img_shape[0] * img_shape[1] * 0.05):
                continue

            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h) if h > 0 else 0
            if not (1.3 < ar < 1.85 or 0.54 < ar < 0.77):
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0

            if solidity < 0.85 or extent < 0.68:
                continue

            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            if len(approx) == 4:
                return approx
            else:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                return np.int64(box).reshape(4, 1, 2)
        return None

    def _four_point_transform(self, image, gray, pts):
        rect = self._order_points(pts.reshape(4, 2))
        
        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_rect = cv2.cornerSubPix(gray, np.float32(rect), (5, 5), (-1, -1), criteria)
        
        # Expansion (scale=0.03)
        rect = self._expand_points(refined_rect, scale=0.03, img_shape=image.shape)
        
        (tl, tr, br, bl) = rect
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        m = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, m, (max_width, max_height))

        return warped

    def _expand_points(self, pts, scale, img_shape):
        center = np.mean(pts, axis=0)
        expanded_pts = []
        h, w = img_shape[:2]
        for i, p in enumerate(pts):
            vec = p - center
            local_scale = scale * 1.2 if i < 2 else scale
            expanded_p = center + vec * (1 + local_scale)
            expanded_p[0] = np.clip(expanded_p[0], 0, w - 1)
            expanded_p[1] = np.clip(expanded_p[1], 0, h - 1)
            expanded_pts.append(expanded_p)
        return np.array(expanded_pts, dtype="float32")

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
