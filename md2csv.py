import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple


def parse_markdown_table(md_path: Path) -> List[Tuple[str, str]]:
    """Parse Markdown table and extract filename and OCR text."""
    results = []
    
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
        writer = csv.writer(f)
        # Write header
        writer.writerow(["filename", "ocr_text"])
        # Write data rows
        for filename, ocr_text in results:
            writer.writerow([filename, ocr_text])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Markdown table to CSV format"
    )
    parser.add_argument(
        "-i", "--input",
        default="results/ocr_results.md",
        help="Input Markdown file",
    )
    parser.add_argument(
        "-o", "--output",
        default="results/ocr_results.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()
    
    md_path = Path(args.input)
    csv_path = Path(args.output)
    
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    
    write_csv_from_markdown(md_path, csv_path)
    print(f"✓ Successfully created: {csv_path}")
    print(f"  Source: {md_path}")
    
    # Display summary
    results = parse_markdown_table(md_path)
    print(f"  Rows: {len(results)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

