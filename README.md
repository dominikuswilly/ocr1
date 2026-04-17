# KTP OCR Information Extraction

High-accuracy NIK extraction from Indonesian KTP images using **PaddleOCR (PP-OCRv4)** and specialized image preprocessing.

## Features
- **PaddleOCR Integration**: Uses the latest server-side models for high precision.
- **Image Enhancement**: Auto-scaling (Lanczos), CLAHE contrast adjustment, and noise reduction.
- **Robust NIK Extraction**: Multi-stage extraction logic (Direct Regex -> Anchor Proximity -> Fallback).
- **Result Persistence**: Automatically saves processed images and extraction metadata (JSON) to the `results/` folder.
- **FastAPI Ready**: Includes a production-ready API structure.

## Prerequisites
- Python 3.10 - 3.12 (Recommended)
- CPU-only `paddlepaddle`

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API

Start the FastAPI server:
```bash
python main.py
```
The server will be available at `http://localhost:8000`.

## API Usage

### Extract NIK from KTP Image
- **Endpoint**: `POST /api/v1/extract/ktp`
- **Payload**: `Multipart/form-data` with `file=@your_ktp.jpg`
- **Response**: JSON containing the extracted NIK, confidence, and paths to saved results.

Access the interactive documentation (Swagger UI) at: `http://localhost:8000/docs`

## Project Structure
- `app/core/`: OCR engine initialization and image preprocessing.
- `app/domain/`: Extraction logic and data models.
- `app/infrastructure/`: Persistence layer (file saving).
- `app/api/`: API routes and request handling.
- `results/`: Directory where processed data is stored.
