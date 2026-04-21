import cv2
import numpy as np
import logging

class ImagePreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process(self, image: np.ndarray) -> dict:
        """
        Applies a series of enhancements and returns a dictionary of every step.
        Optimized for Indonesian KTP OCR.
        """
        if image is None:
            return {}

        steps = {"0_original": image}

        # 1. Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        steps["1_grayscale"] = gray

        # 2. Upscale for better OCR accuracy (Target width 2000px)
        # Previous 1200px was a bit too small for fine NIK text in some resolutions.
        h, w = gray.shape[:2]
        target_w = 2000
        scale = target_w / w
        target_h = int(h * scale)
        
        resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        steps["2_upscaled"] = resized

        # 3. Apply Stronger CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # clipLimit=3.0 helps thin characters (like '1') stand out better.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        steps["3_enhanced_clahe"] = enhanced

        # 4. Sharpening & Denoising
        # Subtle Gaussian blur to remove digitization noise while keeping edges
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # High-pass filter sharpening
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # sharpened = cv2.filter2D(denoised, -1, kernel)
        
        steps["4_final_processed"] = denoised

        return steps
