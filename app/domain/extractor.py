import re
import logging
from typing import List, Dict

class KtpDataExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_nik(self, text_blocks: List[Dict]) -> Dict:
        """
        Parses OCR text blocks to find the 16-digit NIK using proximity and regex.
        """
        # 1. Clean and collect all text
        raw_lines = [b['text'] for b in text_blocks]
        
        # 2. Try direct regex match on all blocks (Waterfall 1: Flat search)
        for block in text_blocks:
            text = block['text'].replace(" ", "")
            match = re.search(r'\d{16}', text)
            if match:
                return {
                    "nik": match.group(0),
                    "confidence": block['confidence'],
                    "method": "direct_regex"
                }

        # 3. Anchor-based search (Waterfall 2: Find "NIK" label)
        # Sometimes the NIK is read as a separate block from the label
        nik_anchor_box = None
        for block in text_blocks:
            if "NIK" in block['text'].upper():
                nik_anchor_box = block['box']
                break
        
        if nik_anchor_box:
            # Find the block closest to the right of the "NIK" label
            best_candidate = None
            min_dist = float('inf')
            
            # anchor_y_center = (nik_anchor_box[0][1] + nik_anchor_box[2][1]) / 2
            # anchor_x_right = nik_anchor_box[1][0]
            
            for block in text_blocks:
                # Skip the anchor itself
                if "NIK" in block['text'].upper(): continue
                
                # Check horizontal alignment (overlap in Y axis)
                # and position to the right
                text = block['text'].replace(" ", "").replace(":", "")
                digits = re.sub(r'\D', '', text)
                
                if len(digits) >= 15: # Allow 15 or 16 in case of minor OCR miss
                    # Heuristic: Same line (y-coordinates similar)
                    if self._is_on_same_line(nik_anchor_box, block['box']):
                        return {
                            "nik": digits[:16].zfill(16),
                            "confidence": block['confidence'],
                            "method": "anchor_proximity"
                        }

        # 4. Fallback: Concatenate all digits and hunt
        all_text = "".join(raw_lines).replace(" ", "")
        all_digits = re.findall(r'\d{16}', all_text)
        if all_digits:
            return {
                "nik": all_digits[0],
                "confidence": 0.5, # Lower confidence for fallback
                "method": "concatenate_fallback"
            }

        return {"nik": None, "confidence": 0.0, "method": "none"}

    def _is_on_same_line(self, box1, box2, tolerance=30):
        # Center Y of both boxes
        y1 = (box1[0][1] + box1[2][1]) / 2
        y2 = (box1[0][1] + box1[2][1]) / 2
        return abs(y1 - y2) < tolerance
