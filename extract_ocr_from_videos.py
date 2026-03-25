import argparse
import re
from pathlib import Path
from typing import List, Tuple

import cv2
from yomitoku import OCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one frame around a target second from each MP4, "
            "run OCR with Yomitoku, and save results in a one-file-per-line CSV."
        )
    )
    parser.add_argument("--input-dir", default="input", help="Directory containing MP4 files")
    parser.add_argument("--output-dir", default="output", help="Directory for output files")
    parser.add_argument("--sec", type=float, default=1.0, help="Target timestamp in seconds")
    parser.add_argument("--lang", default="jpn+eng", help="OCR language hint (Yomitoku expects jpn+eng)")
    parser.add_argument("--device", default="cpu", help="Yomitoku device: cpu or cuda")
    parser.add_argument(
        "--csv-name",
        default="ocr_horizontal.csv",
        help="Output CSV filename inside output-dir",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    cleaned = text.replace("\r", " ").replace("\n", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def sort_choices_in_text(text: str) -> str:
    """選択肢グループ（,③,第3文型 など）を①②③④の昇順に並べ替える。"""
    circled_order = {"①": 1, "②": 2, "③": 3, "④": 4}
    choice_block = re.compile(r"((?:,[①②③④],第[1-4]文型)+)")

    def _sort_block(m: re.Match) -> str:
        pairs = re.findall(r",([①②③④]),(第[1-4]文型)", m.group(1))
        pairs_sorted = sorted(pairs, key=lambda p: circled_order.get(p[0], 99))
        return "".join(f",{circ},{kei}" for circ, kei in pairs_sorted)

    return choice_block.sub(_sort_block, text)


def format_ocr_text_for_csv(text: str) -> str:
    cleaned = normalize_text(text)
    if not re.search(r"[1-4]\s*第\s*[1-4]\s*文型", cleaned):
        return cleaned

    circled = {"1": "①", "2": "②", "3": "③", "4": "④"}

    def _replace_choice(match: re.Match[str]) -> str:
        choice_no = match.group(1)
        sentence_no = match.group(2)
        return f",{circled[choice_no]},第{sentence_no}文型"

    formatted = re.sub(r"([1-4])\s*第\s*([1-4])\s*文型", _replace_choice, cleaned)
    formatted = re.sub(r"\s*,\s*", ",", formatted)
    formatted = re.sub(r"。(?=,)", "。", formatted)
    formatted = re.sub(r"文型\s+([A-Za-z])", r"文型,\1", formatted)
    formatted = re.sub(r"文型(?=[A-Za-z])", "文型,", formatted)
    formatted = sort_choices_in_text(formatted)
    return formatted.strip(" ,")


def save_image(image_path: Path, image) -> None:
    suffix = image_path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for: {image_path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(encoded.tobytes())


def extract_frame_at_second(video_path: Path, sec: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    target_frame = int(max(sec, 0.0) * fps)
    if frame_count > 0:
        target_frame = min(target_frame, frame_count - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to decode frame for video: {video_path}")

    return frame


def crop_image_for_ocr(image, right_crop: int = 100, bottom_crop: int = 20):
    height, width = image.shape[:2]
    cropped_width = max(1, width - max(0, right_crop))
    cropped_height = max(1, height - max(0, bottom_crop))
    return image[:cropped_height, :cropped_width]


def prepare_ocr_input(source_img):
    cropped = crop_image_for_ocr(source_img)
    # Yomitoku expects BGR image input, so convert single-channel image if needed.
    if len(cropped.shape) == 2:
        return cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    return cropped


def run_ocr(ocr_input_img, ocr_model: OCR, lang: str) -> str:
    if lang != "jpn+eng":
        raise ValueError("Yomitoku mode supports --lang jpn+eng only.")

    result, _ = ocr_model(ocr_input_img)
    text = " ".join(word.content for word in result.words if word.content)
    return format_ocr_text_for_csv(text)


def write_vertical_csv(csv_path: Path, ordered_results: List[Tuple[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        f.write("filename,ocr_text\n")
        for filename, text in ordered_results:
            f.write(f"{filename},{text}\n")


def main() -> int:
    args = parse_args()
    ocr_model = OCR(device=args.device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    ocr_input_dir = output_dir / "ocr_input"
    csv_path = output_dir / args.csv_name

    frames_dir.mkdir(parents=True, exist_ok=True)
    ocr_input_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 files found in: {input_dir}")

    results: List[Tuple[str, str]] = []
    for video in videos:
        print(f"Processing: {video.name}")
        frame = extract_frame_at_second(video, args.sec)

        frame_path = frames_dir / f"{video.stem}_{int(args.sec)}s.jpg"
        save_image(frame_path, frame)

        ocr_input = prepare_ocr_input(frame)
        ocr_input_path = ocr_input_dir / f"{video.stem}_{int(args.sec)}s_ocr_input.png"
        save_image(ocr_input_path, ocr_input)

        text = run_ocr(ocr_input, ocr_model, args.lang)
        results.append((video.name, text))

    write_vertical_csv(csv_path, results)
    print(f"Saved CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

