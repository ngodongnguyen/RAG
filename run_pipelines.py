from pathlib import Path
import hashlib
from typing import List, Tuple, Dict, Any
from datetime import date
import re

from split_sections import export_md_sections  # <-- dùng splitter cho .md
from embed_st import embed_chunks_st
from extract_metadata import extract_metadata_batch
from weaviate_client import ensure_schema, upsert_chunks


def _build_kw_texts(metadatas: List[dict]) -> List[str]:
    out = []
    for m in metadatas:
        title = (m or {}).get("title") or ""
        abstract = (m or {}).get("abstract") or ""
        kws = (m or {}).get("keywords") or []
        kw_str = ", ".join([str(k) for k in kws]) if isinstance(kws, list) else str(kws)
        text = ". ".join([s for s in [title.strip(), abstract.strip()] if s])
        if kw_str:
            text = f"{text}. {kw_str}" if text else kw_str
        out.append(text.strip())
    return out

def _normalize_metadatas(chunks: List[str], metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Đảm bảo đủ 4 trường: created_date, title, abstract, keywords."""
    today = date.today().isoformat()
    out = []
    for ch, m in zip(chunks, metadatas):
        m = m if isinstance(m, dict) else {}
        title = (m.get("title") or "").strip()
        abstract = (m.get("abstract") or "").strip()
        kws = m.get("keywords")
        if not isinstance(kws, list):
            kws = [] if kws is None else [str(kws)]
        # Fallback: nếu thiếu title/abstract thì lấy từ nội dung
        if not title:
            # lấy dòng tiêu đề in đậm **...** nếu có, hoặc dòng đầu
            m_title = re.search(r"\*\*([^*]+)\*\*", ch)
            title = (m_title.group(1).strip() if m_title else ch.strip().splitlines()[0])[:200]
        if not abstract:
            # lấy 1–2 dòng đầu sau tiêu đề
            lines = [l.strip() for l in ch.splitlines() if l.strip()]
            abstract = " ".join(lines[1:3])[:400] if len(lines) > 1 else ""
        out.append({
            "created_date": today,
            "title": title,
            "abstract": abstract,
            "keywords": kws
        })
    return out

def run_pipeline_dual(
    input_md: str = "docs/products.md",
    out_sections: str = "docs/output_sections.md",
    class_full: str = "DocumentFull",
    class_kw: str = "DocumentKW"
):
    # 1) Tách SECTION từ .md
    print(f"[1] Tách section từ {input_md} ...")
    chunks, _ = export_md_sections(input_md, out_sections)
    print(f"→ {len(chunks)} section.")

    # 2) Trích metadata
    print(f"[2] Trích metadata ...")
    raw_metas = extract_metadata_batch(chunks)  # có thể thiếu trường
    metadatas = _normalize_metadatas(chunks, raw_metas)
    print(f"→ {len(metadatas)} metadata (đã chuẩn hóa created_date/title/abstract/keywords).")

    # 3) Văn bản keyword
    print(f"[3] Tạo văn bản keyword từ metadata ...")
    kw_texts = _build_kw_texts(metadatas)

    # 4) Embedding FULL
    print(f"[4] Embedding FULL ...")
    full_vecs = embed_chunks_st(chunks)
    print(f"→ {len(full_vecs)} vector FULL.")

    # 5) Embedding KEYWORD
    print(f"[5] Embedding KEYWORD ...")
    kw_vecs = embed_chunks_st(kw_texts)
    print(f"→ {len(kw_vecs)} vector KEYWORD.")

    # 6) Schema (v4)
    print(f"[6] Đảm bảo schema ...")
    ensure_schema(class_full)
    ensure_schema(class_kw)

    # 7) Upsert
    print(f"[7] Upsert FULL → {class_full}")
    ids = [hashlib.md5(c.encode("utf-8")).hexdigest() for c in chunks]
    upsert_chunks(class_full, ids, chunks, [e["embedding"] for e in full_vecs], metadatas)

    print(f"[8] Upsert KEYWORD → {class_kw}")
    upsert_chunks(class_kw, ids, kw_texts, [e["embedding"] for e in kw_vecs], metadatas)

    print("[9] Hoàn tất 2 luồng nhúng.")
if __name__ == "__main__": 
    run_pipeline_dual()