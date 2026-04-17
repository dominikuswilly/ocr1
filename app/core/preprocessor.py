import cv2
import numpy as np
import logging

class ImagePreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process(self, image: np.ndarray) -> dict:
        """
        Applies a series of enhancements and returns a dictionary of every step.
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

        # 2. Resize to a consistent width (e.g., 1200px) keeping aspect ratio
        h, w = gray.shape[:2]
        target_w = 1200
        scale = target_w / w
        target_h = int(h * scale)
        
        resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        steps["2_resized"] = resized

        # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        steps["3_enhanced_clahe"] = enhanced

        # 4. Denoising
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        steps["4_final_processed"] = denoised

        return steps
