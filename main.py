import os
# MUST be set before any paddle modules are imported
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'

import uvicorn
from fastapi import FastAPI
from app.api.routes import router as ktp_router
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="KTP OCR API",
    description="High-accuracy KTP information extraction using PaddleOCR",
    version="1.0.0"
)

# Include routes
app.include_router(ktp_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to KTP OCR API. Go to /docs for Swagger UI."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
