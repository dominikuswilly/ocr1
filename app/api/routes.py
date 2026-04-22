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

            # 4. Extract Data
            if text_blocks:
                nik_data = extractor.extract_nik(text_blocks)
                nama_data = extractor.extract_nama(text_blocks, nik_box=nik_data.get("box"))
                
                # Duplicate processing steps into field-specific folders
                for folder in ["nik", "nama"]:
                    for step_key in ["0_cropped", "1_enhanced", "2_grayscale", "3_denoised", "4_final_processed"]:
                        if step_key in steps:
                            steps[f"{folder}/{step_key}"] = steps[step_key]

                # 4.5 Visualize results
                viz_combined = final_processed.copy()
                if len(viz_combined.shape) == 2:
                    viz_combined = cv2.cvtColor(viz_combined, cv2.COLOR_GRAY2BGR)
                
                # NIK Visualization
                if nik_data.get("nik") and nik_data.get("box"):
                    # Field-specific viz
                    viz_nik = final_processed.copy()
                    if len(viz_nik.shape) == 2: viz_nik = cv2.cvtColor(viz_nik, cv2.COLOR_GRAY2BGR)
                    
                    box = np.array(nik_data["box"], dtype=np.int32)
                    cv2.polylines(viz_nik, [box], True, (0, 255, 0), 2)
                    cv2.polylines(viz_combined, [box], True, (0, 255, 0), 2)
                    
                    label = f"NIK: {nik_data['nik']}"
                    tl = box[np.argmin(box.sum(axis=1))]
                    cv2.putText(viz_nik, label, (tl[0], tl[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(viz_combined, label, (tl[0], tl[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    
                    steps["nik/5_visualized"] = viz_nik

                # Nama Visualization
                if nama_data.get("nama") and nama_data.get("box"):
                    # Field-specific viz
                    viz_nama = final_processed.copy()
                    if len(viz_nama.shape) == 2: viz_nama = cv2.cvtColor(viz_nama, cv2.COLOR_GRAY2BGR)
                    
                    box = np.array(nama_data["box"], dtype=np.int32)
                    cv2.polylines(viz_nama, [box], True, (255, 255, 0), 2)
                    cv2.polylines(viz_combined, [box], True, (255, 255, 0), 2)
                    
                    label = f"nama: {nama_data['nama']}"
                    tl = box[np.argmin(box.sum(axis=1))]
                    cv2.putText(viz_nama, label, (tl[0], tl[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                    cv2.putText(viz_combined, label, (tl[0], tl[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                    
                    steps["nama/5_visualized"] = viz_nama
                
                steps["5_visualized"] = viz_combined

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
            "nama": nama_data.get("nama"),
            "confidence": nik_data.get("confidence"),
            "method": nik_data.get("method"),
            "raw_text": [b["text"] for b in text_blocks],
            "debug_error": debug_error
        }
        saved_path = storage.save_steps(file.filename, steps, metadata)

    return KtpResult(
        nik=nik_data.get("nik"),
        nama=nama_data.get("nama") if text_blocks else None,
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
