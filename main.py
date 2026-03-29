import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, List, Tuple, Optional

import cv2
from yomitoku import OCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one frame around a target second from each MP4, "
            "run OCR with Yomitoku, and save only merged 03+02+01 text."
        )
    )
    parser.add_argument("-o", "--output-dir", default="output", help="Directory for output files")
    parser.add_argument(
        "-i",
        "--input-dir",
        #default="input",
        default="D:\ベリタス動画英語",
        help="Input folder containing videos (subfolders are processed recursively)",
    )
    parser.add_argument("--sec", type=float, default=1.0, help="Target timestamp in seconds")
    parser.add_argument("--lang", default="jpn+eng", help="OCR language hint (Yomitoku expects jpn+eng)")
    parser.add_argument("--device", default="cpu", help="Yomitoku device: cpu or cuda")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    cleaned = text.replace("\r", " ").replace("\n", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def format_reversed_blank_sentence(text: str) -> str:
    pattern = re.compile(
        r"\)\s*([^()]+?)\.\s*([A-Z][A-Za-z'/-]*(?:\s+[A-Za-z'/-]+)*)\s*\("
    )

    def _replace(match: re.Match[str]) -> str:
        tail = match.group(1).strip()
        subject = match.group(2).strip()
        return f"{subject} (          ) {tail}."

    formatted = pattern.sub(_replace, text)
    # OCR sometimes outputs this item as "SMOUS I postponed. ) tomorrow ... If it (".
    # Convert that broken reversed fragment into the expected blank sentence.
    fallback = re.compile(
        r"(?:[A-Za-z'/-]+\s+)?I\s+postponed\.\s*\)\s*tomorrow,\s*the\s+soccer\s+game\s+will\s+be\s*If\s+it\s*\(",
        re.IGNORECASE,
    )
    formatted = fallback.sub(
        "If it (          ) tomorrow, the soccer game will be postponed.",
        formatted,
    )
    # OCR can output "cancelled. If it (...) tomorrow, the picnic will be" in reverse order.
    sentence_tail_first = re.compile(
        r"([A-Za-z][A-Za-z'/-]*\.)\s*(If\s+it\s*\(\s*\)\s*[^.]*?\bwill\s+be)\b",
        re.IGNORECASE,
    )
    formatted = sentence_tail_first.sub(r"\2 \1", formatted)
    return formatted


def normalize_choice_ocr_errors(text: str) -> str:
    """Fix OCR errors in choice blocks: convert 'I' (letter) to '1' (digit) when it appears as a choice number.
    Example: '3 was taking 4 take I took 2 is taking' -> '3 was taking 4 take 1 took 2 is taking'
    """
    # Match blocks with multiple numbered choices like: [1-4] word(s) repeated 2-4 times
    # Within such blocks, replace standalone 'I' followed by lowercase word (choice verb) with '1'
    choice_word = r"[A-Za-z'/-]+(?:\s+[a-z'/-]+){0,5}"
    pattern = re.compile(rf"([234]\s+{choice_word}\s+)I(\s+{choice_word}(?:\s+[234]\s+{choice_word})*)")

    def _replace(m: re.Match[str]) -> str:
        before = m.group(1)
        after = m.group(2)
        return before + "1" + after

    text = pattern.sub(_replace, text)

    # Handle orphaned choice number (e.g., "1 2 will study" -> "1 study 2 will study" or similar pattern)
    # When a number is followed directly by another number, try to synthesize missing choice
    choice_word_loose = r"[A-Za-z'/-]+(?:\s+[a-z'/-]+){0,5}"
    orphan_pattern = re.compile(rf"([1-4])\s+([1-4]\s+{choice_word_loose})")

    def _fix_orphan(m: re.Match[str]) -> str:
        orphan_num = m.group(1)
        next_part = m.group(2)
        # Extract the number and full verb phrase from next_part
        # e.g., "2 will study" -> take "will study" for choice 1, and keep "2 will study" as is
        match_next = re.match(rf"([1-4])\s+({choice_word_loose})", next_part)
        if match_next:
            next_num = match_next.group(1)
            next_verb = match_next.group(2)
            # Use the full verb phrase for the orphaned choice
            return f"{orphan_num} {next_verb} {next_part}"
        return m.group(0)

    text = orphan_pattern.sub(_fix_orphan, text)
    return text



def _remove_tense_choice_noise(text: str, choice_word: str) -> str:
    """Remove a single noisy capitalized token between numbered tense choices."""
    return re.sub(
        rf"([1-4]\s+{choice_word})\s+[A-Z][a-z]*\s+([1-4]\s)",
        r"\1 \2",
        text,
    )


def _parse_tense_choice_pairs(block: str, choice_word: str) -> List[Tuple[str, str]]:
    return re.findall(rf"([1-4])\s+({choice_word})", block)


def _format_tense_choice_block(block: str, choice_word: str, circled: dict) -> Optional[str]:
    pairs = _parse_tense_choice_pairs(block, choice_word)
    if not pairs:
        return None
    # Skip broken OCR blocks that reuse the same choice number.
    if len({no for no, _ in pairs}) != len(pairs):
        return None
    pairs_sorted = sorted(pairs, key=lambda p: int(p[0]))
    return "," + ",".join(f"{circled[no]},{word}" for no, word in pairs_sorted) + ","


def format_tense_choices(text: str) -> str:
    """'1 have you finished 2 did you finish ...' のような多語選択肢を番号付きで整形する。"""
    circled = {"1": "①", "2": "②", "3": "③", "4": "④"}
    # Allow uppercase on the first token only (Do/Did/Have), then keep following tokens lowercase
    # to avoid swallowing sentence heads like "When".
    choice_word = r"[A-Za-z'/-]+(?:\s+[a-z'/-]+){0,5}"

    normalized = _remove_tense_choice_noise(text, choice_word)
    pattern = re.compile(rf"\s*((?:[1-4]\s+{choice_word}\s*){{2,4}})")

    def _replace(match: re.Match) -> str:
        block = match.group(1)
        formatted = _format_tense_choice_block(block, choice_word, circled)
        return formatted if formatted is not None else match.group(0)

    return pattern.sub(_replace, normalized)


def sort_choices_in_text(text: str) -> str:
    """選択肢グループ（,③,第3文型 など）を①②③④の昇順に並べ替える。文型は1～5対応。"""
    circled_order = {"①": 1, "②": 2, "③": 3, "④": 4}
    choice_block = re.compile(r"((?:,[①②③④],第[1-5]文型)+)")

    def _sort_block(m: re.Match) -> str:
        pairs = re.findall(r",([①②③④]),(第[1-5]文型)", m.group(1))
        pairs_sorted = sorted(pairs, key=lambda p: circled_order.get(p[0], 99))
        return "".join(f",{circ},{kei}" for circ, kei in pairs_sorted)

    return choice_block.sub(_sort_block, text)


def _third_person_s(verb: str) -> str:
    lower = verb.lower()
    if lower.endswith("y") and len(lower) >= 2 and lower[-2] not in "aeiou":
        return lower[:-1] + "ies"
    if lower.endswith(("s", "sh", "ch", "x", "z", "o")):
        return lower + "es"
    return lower + "s"


def recover_missing_choice_one(text: str) -> str:
    """Fill missing ① from common tense-option patterns when OCR drops choice 1."""
    if ",①," in text:
        return text

    pattern = re.compile(r",②,will\s+([A-Za-z'/-]+),③,([A-Za-z'/-]+),④,([A-Za-z'/-]+),")

    def _replace(m: re.Match[str]) -> str:
        base2 = m.group(1).lower()
        base3 = m.group(2).lower()
        ing = m.group(3).lower()
        # Typical set: will snow / snow / snowing  -> missing snows
        if base2 == base3 and ing in {base3 + "ing", (base3[:-1] + "ing") if base3.endswith("e") else ""}:
            return f",①,{_third_person_s(base3)}" + m.group(0)
        return m.group(0)

    return pattern.sub(_replace, text)


def format_choices_for_split02(text: str) -> str:
    """Format raw OCR choice text for _02.txt log.
    Example: '1 l am 2 do I 3 am I 4 did I' -> '①, I am, ②, do I, ③, am I, ④, did I'
    """
    cleaned = normalize_text(text)
    if cleaned and not cleaned.lstrip().startswith("1"):
        # 02.txt starts with choice 1; if missing, replace the first non-space character with "1".
        cleaned = re.sub(r"\S", "1", cleaned, count=1)
    cleaned = normalize_choice_ocr_errors(cleaned)

    # Strict path for fixed order: 1 ... 2 ... 3 ... 4 ...
    fixed = re.search(r"\b1\s+(.+?)\s+2\s+(.+?)\s+3\s+(.+?)\s+4\s+(.+)$", cleaned)
    if fixed:
        c1, c2, c3, c4 = (part.strip(" ,") for part in fixed.groups())
        # Common OCR confusion for the pronoun "I" at choice head.
        c1 = re.sub(r"^[lI]\s+", "I ", c1)
        return f"①, {c1}, ②, {c2}, ③, {c3}, ④, {c4}".strip(" ,")

    # Fallback for unexpected patterns.
    return cleaned


def build_csv_source_from_splits(split_raw_texts: List[str]) -> str:
    """Build CSV source text in fixed order: _03 + _02 + _01."""
    if not split_raw_texts:
        return ""

    # Keep existing behavior when split count is unexpected.
    if len(split_raw_texts) < 3:
        return " ".join(t for t in reversed(split_raw_texts) if t).strip()

    part01 = normalize_text(split_raw_texts[0])
    part02 = format_choices_for_split02(split_raw_texts[1])
    part03 = normalize_text(split_raw_texts[2])
    return " ".join(p for p in (part03, part02, part01) if p).strip()


def build_merged_text_from_splits(split_raw_texts: List[str]) -> str:
    """Build plain merged text in fixed order: _03,_02,_01."""
    if not split_raw_texts:
        return ""

    # Keep existing behavior when split count is unexpected.
    if len(split_raw_texts) < 3:
        return ",".join(normalize_text(t) for t in reversed(split_raw_texts) if t).strip(" ,")

    part01 = normalize_text(split_raw_texts[0])
    part02 = format_choices_for_split02(split_raw_texts[1])
    part03 = normalize_text(split_raw_texts[2])
    return ",".join(p for p in (part03, part02, part01) if p).strip(" ,")


def format_ocr_text_for_csv(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = format_reversed_blank_sentence(cleaned)
    cleaned = normalize_choice_ocr_errors(cleaned)
    cleaned = format_tense_choices(cleaned)
    cleaned = recover_missing_choice_one(cleaned)
    if not re.search(r"[1-4]\s*第\s*[1-5]\s*文型", cleaned):
        return cleaned

    circled = {"1": "①", "2": "②", "3": "③", "4": "④"}

    def _replace_choice(match: re.Match[str]) -> str:
        choice_no = match.group(1)
        sentence_no = match.group(2)
        return f",{circled[choice_no]},第{sentence_no}文型"

    formatted = re.sub(r"([1-4])\s*第\s*([1-5])\s*文型", _replace_choice, cleaned)
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


def split_image_into_three_vertical_parts(image) -> List:
    """Split image into top/middle/bottom thirds for OCR."""
    height = image.shape[0]
    if height <= 0:
        return [image]

    cut1 = max(1, height // 3)
    cut2 = max(cut1 + 1, (2 * height) // 3)
    cut2 = min(cut2, height)

    parts = [image[:cut1, :], image[cut1:cut2, :], image[cut2:, :]]
    # Ensure all parts are non-empty to avoid OCR runtime issues.
    return [part for part in parts if part.size > 0]


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(value).items()}
    return str(value)


def _extract_word_text(word: Any) -> str:
    for attr in ("content", "text", "value"):
        value = getattr(word, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_word_xy(word: Any) -> Tuple[float, float]:
    def _from_points(points: Any) -> Optional[Tuple[float, float]]:
        if not points:
            return None
        xs: List[float] = []
        ys: List[float] = []
        for p in points:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
            elif hasattr(p, "x") and hasattr(p, "y"):
                xs.append(float(getattr(p, "x")))
                ys.append(float(getattr(p, "y")))
        if xs and ys:
            return min(xs), min(ys)
        return None

    for attr in ("polygon", "points", "quad", "vertices"):
        pt = _from_points(getattr(word, attr, None))
        if pt is not None:
            return pt

    bbox = getattr(word, "bbox", None)
    if bbox is not None:
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            return float(bbox[0]), float(bbox[1])
        x = getattr(bbox, "x", None)
        y = getattr(bbox, "y", None)
        if x is not None and y is not None:
            return float(x), float(y)
        x1 = getattr(bbox, "x1", None)
        y1 = getattr(bbox, "y1", None)
        if x1 is not None and y1 is not None:
            return float(x1), float(y1)

    return 0.0, 0.0


def build_natural_order_text(ocr_result: Any) -> str:
    words = getattr(ocr_result, "words", None) or []
    entries: List[Tuple[float, float, str]] = []
    for word in words:
        text = _extract_word_text(word).replace("\n", "").replace("\r", "").strip()
        if not text:
            continue
        x, y = _extract_word_xy(word)
        entries.append((x, y, text))

    if not entries:
        return ""

    # Sort by y then x, then merge words line-by-line with a tolerance to preserve reading order.
    entries.sort(key=lambda item: (item[1], item[0]))
    ys = sorted(item[1] for item in entries)
    if len(ys) >= 2:
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        gaps = [g for g in gaps if g > 0]
        tol = max(8.0, (sum(gaps) / len(gaps)) * 0.7) if gaps else 8.0
    else:
        tol = 8.0

    lines: List[List[Tuple[float, str]]] = []
    current_line: List[Tuple[float, str]] = []
    current_y = entries[0][1]
    for x, y, text in entries:
        if not current_line or abs(y - current_y) <= tol:
            current_line.append((x, text))
            current_y = (current_y + y) / 2.0
            continue
        lines.append(current_line)
        current_line = [(x, text)]
        current_y = y
    if current_line:
        lines.append(current_line)

    ordered_tokens: List[str] = []
    for line in lines:
        line.sort(key=lambda item: item[0])
        ordered_tokens.extend(token for _, token in line)
    return " ".join(ordered_tokens)


def save_layout_json(json_path: Path, ocr_result: Any, raw_text: str) -> None:
    payload = {
        "raw_text": raw_text,
        "layout": _to_jsonable(ocr_result),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_ocr(ocr_input_img, ocr_model: OCR, lang: str) -> Tuple[str, Any]:
    if lang != "jpn+eng":
        raise ValueError("Yomitoku mode supports --lang jpn+eng only.")

    result, _ = ocr_model(ocr_input_img)
    # Build text in a stable reading order from layout coordinates.
    raw_text = build_natural_order_text(result)
    return raw_text, result


def write_vertical_csv(csv_path: Path, ordered_results: List[Tuple[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        f.write("filename,ocr_text\n")
        for filename, text in ordered_results:
            clean_text = text.replace('"', '')
            f.write(f'{filename},{clean_text}\n')


def write_text_file(txt_path: Path, text: str) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(text)


def initialize_merged_csv(csv_path: Path) -> None:
    """Create merged CSV with header (UTF-8 BOM) for Excel compatibility."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        f.write("filename,ocr_text\n")


def append_merged_csv_row(csv_path: Path, filename: str, text: str) -> None:
    """Append one OCR row to merged CSV."""
    clean_filename = filename.replace('"', "")
    clean_text = text.replace('"', "")
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        f.write(f"{clean_filename},{clean_text}\n")


def write_markdown_output(md_path: Path, ordered_results: List[Tuple[str, str, Optional[Path]]], include_figure: bool = False) -> None:
    """Write OCR results in Markdown table format with optional figure embedding."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# OCR Results\n\n")
        f.write("| Filename | OCR Text |\n")
        f.write("|----------|----------|\n")
        for filename, text, img_path in ordered_results:
            # Escape pipe characters in text for Markdown table
            safe_text = text.replace("|", "\\|")
            f.write(f"| {filename} | {safe_text} |\n")
            
            # Include figure if requested and image path is available
            if include_figure and img_path and img_path.exists():
                f.write(f"\n![{filename}]({img_path.relative_to(md_path.parent)})\n\n")


def parse_markdown_table(md_path: Path) -> List[Tuple[str, str]]:
    """Parse Markdown table and extract filename and OCR text."""
    results = []
    
    if not md_path.exists():
        return results
    
    with md_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Find table rows (lines that start and end with |)
    for line in lines:
        line = line.strip()
        
        # Skip non-table lines, headers, separators
        if not line.startswith("|") or not line.endswith("|"):
            continue
        
        # Skip separator line (contains dashes)
        if re.match(r"^\|\s*[-\s|]+\s*$", line):
            continue
        
        # Skip header line (contains "Filename" and "OCR Text")
        if "Filename" in line and "OCR Text" in line:
            continue
        
        # Parse the table row
        # Format: | filename | ocr_text |
        parts = [cell.strip() for cell in line.split("|")]
        # Remove empty first and last elements
        parts = [p for p in parts if p]
        
        if len(parts) >= 2:
            filename = parts[0]
            ocr_text = parts[1]
            
            # Unescape pipe characters
            ocr_text = ocr_text.replace(r"\|", "|")
            
            results.append((filename, ocr_text))
    
    return results


def write_csv_from_markdown(md_path: Path, csv_path: Path) -> None:
    """Convert Markdown table to CSV."""
    results = parse_markdown_table(md_path)
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        f.write("filename,ocr_text\n")
        for filename, ocr_text in results:
            clean_text = ocr_text.replace('"', '')
            f.write(f'{filename},{clean_text}\n')


def clear_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def find_videos_recursively(input_dir: Path) -> List[Path]:
    # Case-insensitive search for .mp4 in all subdirectories.
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4")


def build_output_stem(video_path: Path, input_dir: Path) -> str:
    # Keep output names stable and unique across nested folders.
    rel_no_ext = video_path.relative_to(input_dir).with_suffix("")
    return "__".join(rel_no_ext.parts)


def main(sec: Optional[float] = None, folder: Optional[str] = None) -> int:
    args = parse_args()
    target_sec = args.sec if sec is None else sec
    special_video_name = "標準演習_文型_5_6_part_004.mp4"
    special_video_sec = 3.0
    ocr_model = OCR(device=args.device)

    # Input videos are scanned recursively from the selected folder.
    input_dir = Path(folder) if folder else Path(args.input_dir)
    output_dir = Path(args.output_dir)
    raw_text_dir = output_dir / "raw_text"

    clear_output_dir(output_dir)
    raw_text_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos_recursively(input_dir)
    if not videos:
        raise FileNotFoundError(f"No MP4 files found in (recursive): {input_dir}")

    merged_csv_path = raw_text_dir / "merged_all.csv"
    initialize_merged_csv(merged_csv_path)

    for video in videos:
        rel_video = video.relative_to(input_dir)
        print(f"Processing: {rel_video.as_posix()}")
        sec_for_video = special_video_sec if video.name == special_video_name else target_sec
        frame = extract_frame_at_second(video, sec_for_video)

        ocr_input = prepare_ocr_input(frame)

        split_inputs = split_image_into_three_vertical_parts(ocr_input)
        split_raw_texts: List[str] = []
        for split_img in split_inputs:
            split_raw, _ = run_ocr(
                split_img,
                ocr_model,
                args.lang,
            )
            split_raw_texts.append(split_raw)
        merged_text = build_merged_text_from_splits(split_raw_texts)
        append_merged_csv_row(merged_csv_path, rel_video.as_posix(), merged_text)
        print(f"Appended CSV row: {rel_video.as_posix()}")

    print(f"Saved merged CSV: {merged_csv_path}")
        
    return 0


if __name__ == "__main__":
    # Change this value to process a different root folder.
    #raise SystemExit(main(sec=1.0, folder="D:\ベリタス動画英語"))
    raise SystemExit(main(sec=1.0, folder="input"))
