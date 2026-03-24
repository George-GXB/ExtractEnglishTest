# ExtractEnglishTest

Extract one frame around 1 second from each MP4 in `input`, binarize the frame,
run OCR with Yomitoku, and save results as a one-file-per-line CSV.

## Requirements

- Python 3.10+
- Yomitoku dependencies (installed via pip)

Python dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python extract_ocr_from_videos.py --input-dir input --output-dir output --sec 1 --lang jpn+eng --device cpu
```

## Output

- `output/frames/*_1s.jpg`: extracted frame images
- `output/binary/*_1s_binary.png`: binarized images
- `output/ocr_input/*_1s_ocr_input.png`: images actually passed to Yomitoku OCR (cropped 20px from right and bottom)
- `output/ocr_horizontal.csv`: CSV (one file per line)

CSV format:

- Header: `filename,ocr_text`
- Data rows: one video per line

