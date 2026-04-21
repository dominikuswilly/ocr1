from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse
import os
import cv2
import numpy as np
import io
import logging
from PIL import Image

from app.core.ocr_engine import KtpOcrEngine
from app.core.preprocessor import ImagePreprocessor
from app.core.detector import KtpDetector
from app.domain.extractor import KtpDataExtractor
from app.infrastructure.storage import ResultStorage
from app.domain.models import KtpResult

router = APIRouter()

# Initialize services
ocr_engine = KtpOcrEngine()
detector = KtpDetector()
preprocessor = ImagePreprocessor()
extractor = KtpDataExtractor()
storage = ResultStorage()

@router.post("/extract/ktp", response_model=KtpResult)
async def extract_ktp(file: UploadFile = File(...)):
    # 1. Read file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    debug_error = None
    text_blocks = []
    nik_data = {"nik": None, "confidence": 0.0, "method": "none"}
    steps = {}

    try:
        # 2. Card Detection/Cropping
        cropped_img = detector.detect_and_crop(img)
        steps["0_cropped"] = cropped_img

        # 3. Preprocess the cropped image (returns a dictionary of steps)
        prep_steps = preprocessor.process(cropped_img)
        steps.update(prep_steps)
        final_processed = steps.get("4_final_processed")

        if final_processed is not None:
            # 3. Perform OCR
            text_blocks = ocr_engine.extract_text_blocks(final_processed)

            # 4. Extract NIK
            if text_blocks:
                nik_data = extractor.extract_nik(text_blocks)
                
                # 4.5 Visualize the result on a diagnostic image
                if nik_data.get("nik") and nik_data.get("box"):
                    viz_img = final_processed.copy()
                    # Convert to BGR for color drawing if grayscale
                    if len(viz_img.shape) == 2:
                        viz_img = cv2.cvtColor(viz_img, cv2.COLOR_GRAY2BGR)
                    
                    # Draw ROI box
                    box = np.array(nik_data["box"], dtype=np.int32)
                    cv2.polylines(viz_img, [box], True, (0, 255, 0), 3)
                    
                    # Draw text label
                    label = f"NIK: {nik_data['nik']}"
                    # Find top-left most point for label placement
                    tl = box[np.argmin(box.sum(axis=1))]
                    cv2.putText(viz_img, label, (tl[0], tl[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    steps["5_visualized"] = viz_img

            else:
                debug_error = f"OCR engine returned no text blocks. Engine error: {ocr_engine.last_error}"
        else:
            debug_error = "Preprocessing failed to produce a final image."

    except Exception as e:
        debug_error = f"Pipeline failure: {str(e)}"
        import traceback
        logging.error(traceback.format_exc())

    # 5. Save Results (including all steps)
    saved_path = None
    if steps:
        metadata = {
            "filename": file.filename,
            "nik": nik_data.get("nik"),
            "confidence": nik_data.get("confidence"),
            "method": nik_data.get("method"),
            "raw_text": [b["text"] for b in text_blocks],
            "debug_error": debug_error
        }
        saved_path = storage.save_steps(file.filename, steps, metadata)

    return KtpResult(
        nik=nik_data.get("nik"),
        confidence=nik_data.get("confidence"),
        extraction_method=nik_data.get("method"),
        filename=file.filename,
        processed_image_path=saved_path,
        raw_text=[b["text"] for b in text_blocks],
        debug_error=debug_error
    )

@router.post("/detect/ktp")
async def detect_ktp(file: UploadFile = File(...)):
    # 1. Read file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        # 2. Detect and Crop
        cropped_img = detector.detect_and_crop(img)
        
        if cropped_img is None:
            raise HTTPException(status_code=422, detail="KTP not detected in image")

        # 3. Save as a standalone result
        # We reuse save_steps but just for one step
        steps = {"0_focused": cropped_img}
        metadata = {"filename": file.filename, "type": "detection_only"}
        saved_path = storage.save_steps(file.filename, steps, metadata)

        # 4. Return the file as a response
        return FileResponse(
            path=saved_path,
            media_type="image/jpeg",
            filename=f"detected_{file.filename}"
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Detection failure: {str(e)}")
