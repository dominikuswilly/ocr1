import os
import json
import cv2
import numpy as np
import logging
from datetime import datetime

class ResultStorage:
    def __init__(self, base_dir: str = "results"):
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def save_steps(self, filename: str, steps: dict, metadata: dict) -> str:
        """
        Saves all intermediate processed images and the extraction metadata.
        Returns the path to the final processed image.
        """
        # Create a unique timestamp for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = filename.replace(".", "_")
        session_dir = os.path.join(self.base_dir, f"{timestamp}_{safe_filename}")
        
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)

        final_img_path = None
        # Save All Image Steps
        for step_name, image in steps.items():
            img_path = os.path.join(session_dir, f"{step_name}.jpg")
            # Ensure subdirectory exists if step_name contains '/'
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            cv2.imwrite(img_path, image)
            if "final" in step_name and "/" not in step_name:
                final_img_path = img_path

        # Save Metadata (JSON)
        json_path = os.path.join(session_dir, "extraction_result.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        self.logger.info(f"Results saved to {session_dir}")
        return final_img_path or os.path.join(session_dir, "4_final_processed.jpg")
