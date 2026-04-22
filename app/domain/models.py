from pydantic import BaseModel
from typing import Optional, List

class KtpResult(BaseModel):
    nik: Optional[str] = None
    nama: Optional[str] = None
    province: Optional[str] = None
    confidence: float = 0.0
    extraction_method: str = "direct"
    filename: str
    processed_image_path: Optional[str] = None
    raw_text: Optional[List[str]] = None
    debug_error: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str
