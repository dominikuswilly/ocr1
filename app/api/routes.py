from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.ocr_engine import KtpOcrEngine
from app.core.preprocessor import ImagePreprocessor
from app.domain.extractor import KtpDataExtractor
from app.infrastructure.storage import ResultStorage
from app.domain.models import KtpResult
import cv2
import numpy as np
import io
from PIL import Image

router = APIRouter()

# Initialize services
ocr_engine = KtpOcrEngine()
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
        # 2. Preprocess (returns a dictionary of steps)
        steps = preprocessor.process(img)
        final_processed = steps.get("4_final_processed")

        if final_processed is not None:
            # 3. Perform OCR
            text_blocks = ocr_engine.extract_text_blocks(final_processed)

            # 4. Extract NIK
            if text_blocks:
                nik_data = extractor.extract_nik(text_blocks)
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
