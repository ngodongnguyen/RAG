from pathlib import Path
from typing import List, Tuple

def split_sections(text: str) -> List[str]:
    """Tách văn bản theo '\n\n\n', mỗi SECTION = 1 chunk."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [s.strip() for s in t.split("\n\n\n") if s.strip()]
    return parts

def export_sections(input_path: str, output_txt: str) -> Tuple[List[str], Path]:
    src = Path(input_path); dst = Path(output_txt)
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = src.read_text(encoding="utf-8")
    sections = split_sections(raw)
    with dst.open("w", encoding="utf-8") as f:
        for i, sec in enumerate(sections, 1):
            f.write(f"\n===== SECTION {i} =====\n{sec}\n")
    return sections, dst
