from pathlib import Path
from typing import List, Tuple
import re

def split_md_sections(text: str) -> List[str]:
    """
    Tách văn bản .md thành từng SECTION.
    Mỗi SECTION bắt đầu bằng dòng có pattern **Tên sản phẩm**.
    """
    # Chuẩn hóa xuống dòng
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Tách theo pattern Markdown tiêu đề sản phẩm (**Tên**)
    # Giữ lại tiêu đề trong kết quả
    parts = re.split(r"\n(?=\*\*[^*]+?\*\*)", t)
    sections = [p.strip() for p in parts if p.strip()]
    return sections


def export_md_sections(input_path: str, output_md: str) -> Tuple[List[str], Path]:
    """
    Đọc file .md và xuất ra file mới trong đó mỗi SECTION được tách rõ ràng:
    ===== SECTION N =====
    """
    src = Path(input_path); dst = Path(output_md)
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw = src.read_text(encoding="utf-8")
    sections = split_md_sections(raw)

    with dst.open("w", encoding="utf-8") as f:
        for i, sec in enumerate(sections, 1):
            f.write(f"\n===== SECTION {i} =====\n{sec}\n")

    return sections, dst
