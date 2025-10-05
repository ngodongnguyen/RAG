from pathlib import Path
import hashlib
from typing import List, Tuple

from split_sections import export_sections
from embed_st import embed_chunks_st
from extract_metadata import extract_metadata_batch
from weaviate_client import ensure_schema, upsert_chunks


def _build_kw_texts(metadatas: List[dict]) -> List[str]:
    """
    Tạo văn bản ngắn cho luồng keyword từ metadata:
    "<title>. <abstract>. kw1, kw2, kw3"
    """
    out = []
    for m in metadatas:
        title = (m or {}).get("title") or ""
        abstract = (m or {}).get("abstract") or ""
        kws = (m or {}).get("keywords") or []
        kw_str = ", ".join([str(k) for k in kws]) if isinstance(kws, list) else str(kws)
        text = ". ".join([s for s in [title.strip(), abstract.strip()] if s])  # "title. abstract"
        if kw_str:
            text = f"{text}. {kw_str}" if text else kw_str
        out.append(text.strip())
    return out


def run_pipeline_dual(
    input_txt: str = "docs/data.txt",
    out_sections: str = "docs/output_sections.txt",
    weaviate_url: str = "http://localhost:8080",
    class_full: str = "DocumentFull",
    class_kw: str = "DocumentKW"
):
    # 1) Tách section
    print(f"[1] Tách section từ {input_txt} ...")
    chunks, _ = export_sections(input_txt, out_sections)
    print(f"→ {len(chunks)} section.")

    # 2) Trích metadata (để lấy title/abstract/keywords cho luồng keyword)
    print(f"[2] Trích metadata ...")
    metadatas = extract_metadata_batch(chunks)
    print(f"→ {len(metadatas)} metadata.")

    # 3) Chuẩn bị văn bản cho luồng keyword
    print(f"[3] Tạo văn bản keyword từ metadata ...")
    kw_texts = _build_kw_texts(metadatas)

    # 4) Embedding ST cho full text
    print(f"[4] Embedding FULL (Sentence-Transformers) ...")
    full_vecs = embed_chunks_st(chunks)  # [{embedding: [...], n_dims: int}, ...]
    print(f"→ {len(full_vecs)} vector FULL.")

    # 5) Embedding ST cho keyword texts
    print(f"[5] Embedding KEYWORD (Sentence-Transformers) ...")
    kw_vecs = embed_chunks_st(kw_texts)
    print(f"→ {len(kw_vecs)} vector KEYWORD.")

    # 6) Kết nối và đảm bảo schema cho 2 lớp
    print(f"[6] Đảm bảo schema trên Weaviate ...")
    ensure_schema(weaviate_url, class_full)
    ensure_schema(weaviate_url, class_kw)

    # 7) Upsert 2 luồng vào 2 lớp khác nhau
    print(f"[7] Upsert FULL → {class_full}")
    ids = [hashlib.md5(c.encode("utf-8")).hexdigest() for c in chunks]
    upsert_chunks(
        weaviate_url=weaviate_url,
        class_name=class_full,
        ids=ids,
        texts=chunks,
        embeddings=[e["embedding"] for e in full_vecs],
        metadatas=metadatas
    )

    print(f"[8] Upsert KEYWORD → {class_kw}")
    upsert_chunks(
        weaviate_url=weaviate_url,
        class_name=class_kw,
        ids=ids,  # cùng id để dễ đối chiếu giữa 2 lớp
        texts=kw_texts,
        embeddings=[e["embedding"] for e in kw_vecs],
        metadatas=metadatas
    )

    print("[9] Hoàn tất 2 luồng nhúng.")


if __name__ == "__main__":
    run_pipeline_dual()
