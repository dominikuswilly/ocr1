import os
import numpy as np

# Performance/Compatibility Flags for PaddlePaddle 3.3.1 on CPU/Windows
# MUST be set before importing paddle/paddleocr
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'

from paddleocr import PaddleOCR
from PIL import Image
import logging
import cv2

class KtpOcrEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_error = None
        
        try:
            # Initialize PaddleOCR with server models for high accuracy
            # use_textline_orientation=True replaces use_angle_cls=True in v3+
            self.ocr = PaddleOCR(
                use_textline_orientation=True, 
                lang='en', 
                ocr_version='PP-OCRv4',
                device='cpu',
                enable_mkldnn=False
            )
            self.logger.info("PaddleOCR engine initialized successfully (v3 API, oneDNN disabled)")
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise e

    def extract_text_blocks(self, image: np.ndarray):
        """
        Performs OCR on the provided image and returns raw text blocks with coordinates.
        """
        try:
            # PaddleOCR 3.x (PaddleX) internal processors (like Normalize) 
            # expect 3-channel images. If image is grayscale (2D), convert to BGR.
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
            # In PaddleOCR 3.x, use_textline_orientation=True replaces cls=True 
            result = self.ocr.ocr(image, use_textline_orientation=True)
            
            text_blocks = []
            if result and len(result) > 0:
                # In PaddleOCR 3.x, result[0] is an OCRResult object
                res = result[0]
                
                # Robust attribute/key extraction
                def get_data(obj, key):
                    if isinstance(obj, dict): return obj.get(key, [])
                    return getattr(obj, key, [])

                texts = get_data(res, 'rec_texts')
                scores = get_data(res, 'rec_scores')
                boxes = get_data(res, 'dt_polys')
                
                if texts and boxes:
                    for text, score, box in zip(texts, scores, boxes):
                        text_blocks.append({
                            "box": box.tolist() if hasattr(box, 'tolist') else box,
                            "text": str(text),
                            "confidence": float(score)
                        })
                else:
                    # Fallback for 2.x structure (result[0] is a list of lines)
                    if isinstance(res, list):
                        for line in res:
                            if len(line) == 2:
                                box = line[0]
                                text, conf = line[1]
                                text_blocks.append({
                                    "box": box,
                                    "text": text,
                                    "confidence": float(conf)
                                })
            
            if not text_blocks:
                self.logger.warning("OCR Engine returned no text blocks.")
                
            return text_blocks
        except Exception as e:
            self.last_error = f"Extraction execution error: {str(e)}"
            self.logger.error(self.last_error)
            import traceback
            self.logger.error(traceback.format_exc())
            return []
        
