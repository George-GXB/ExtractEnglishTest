# ExtractEnglishTest

Extract one frame around 1 second from each MP4 in `input`,
run OCR with Yomitoku, and save results as a one-file-per-line CSV.
Optionally, run additional OCR engines and save each result to a separate CSV.

## Requirements

- Python 3.10+
- Yomitoku dependencies (installed via pip)
- Tesseract OCR executable (optional, required when `--other-ocr tesseract`)
- EasyOCR dependency (optional, required when `--other-ocr easyocr`)

Python dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python extract_ocr_from_videos.py --input-dir input --output-dir output --sec 1 --lang jpn+eng --device cpu
```

Run with additional OCR (Tesseract) and save separate CSV:

```bash
python extract_ocr_from_videos.py --input-dir input --output-dir output --sec 1 --lang jpn+eng --device cpu --other-ocr tesseract
```

Run with multiple additional OCR engines:

```bash
python extract_ocr_from_videos.py --input-dir input --output-dir output --sec 1 --lang jpn+eng --device cpu --other-ocr tesseract,easyocr
```

## Output

- `output/frames/*_1s.jpg`: extracted frame images
- `output/ocr_input/*_1s_ocr_input.png`: images actually passed to Yomitoku OCR (cropped 20px from right and bottom)
- `output/ocr_horizontal.csv`: Yomitoku CSV (one file per line)
- `output/ocr_horizontal_tesseract.csv`: Tesseract CSV (created when `--other-ocr tesseract`)
- `output/ocr_horizontal_easyocr.csv`: EasyOCR CSV (created when `--other-ocr easyocr`)

CSV format:

- Header: `filename,ocr_text`
- Data rows: one video per line

