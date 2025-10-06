# weaviate_client.py — v4, BYOV, RFC3339 date, batch fixed_size
from typing import List, Tuple
import re
import weaviate
from weaviate.classes.config import Property, DataType, Configure


def _connect_local():
    return weaviate.connect_to_local()


def _make_vector_config() -> Tuple[str, object]:
    # v4.16+: Configure.Vectors.self_provided(); cũ: Vectorizer.none()
    if hasattr(Configure, "Vectors"):
        return ("vector_config", Configure.Vectors.self_provided())
    return ("vectorizer_config", Configure.Vectorizer.none())


_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
def _to_rfc3339(date_str: str) -> str:
    if isinstance(date_str, str) and _DATE_ONLY.match(date_str):
        return f"{date_str}T00:00:00Z"
    return date_str


def _normalize_metadata(meta: dict) -> dict:
    if not isinstance(meta, dict):
        return meta
    out = dict(meta)
    if "created_date" in out:
        out["created_date"] = _to_rfc3339(out["created_date"])
    return out


def ensure_schema(class_name: str):
    client = _connect_local()
    try:
        if client.collections.exists(class_name):
            return

        key, vec_cfg = _make_vector_config()
        client.collections.create(
            name=class_name,
            **{key: vec_cfg},
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(
                    name="metadata",
                    data_type=DataType.OBJECT,
                    nested_properties=[
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="abstract", data_type=DataType.TEXT),
                        Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                        Property(name="created_date", data_type=DataType.DATE),
                    ],
                ),
            ],
        )
    finally:
        client.close()


def upsert_chunks(
    class_name: str,
    ids: List[str],
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[dict],
    batch_size: int = 64,
):
    assert len(ids) == len(texts) == len(embeddings) == len(metadatas)

    metas = [_normalize_metadata(m) for m in metadatas]

    client = _connect_local()
    try:
        col = client.collections.get(class_name)
        # dùng fixed_size để set batch_size; dynamic() không có tham số
        with col.batch.fixed_size(size=batch_size) as batch:
            for i in range(len(ids)):
                batch.add_object(
                    properties={"text": texts[i], "metadata": metas[i]},
                    uuid=ids[i],
                    vector=embeddings[i],
                )
    finally:
        client.close()
