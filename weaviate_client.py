from typing import List
from weaviate import Client


def ensure_schema(weaviate_url: str, class_name: str):
    """
    Schema tối giản cho mỗi lớp:
      - vectorizer: none (dùng vector custom)
      - properties:
          text: text
          metadata: object
    """
    client = Client(weaviate_url)
    schema = client.schema.get()
    if any(c.get("class") == class_name for c in schema.get("classes", [])):
        return

    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "metadata", "dataType": ["object"]},
        ],
    }
    client.schema.create_class(class_obj)


def upsert_chunks(
    weaviate_url: str,
    class_name: str,
    ids: List[str],
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[dict],
    batch_size: int = 64
):
    client = Client(weaviate_url)
    n = len(ids)
    assert n == len(texts) == len(embeddings) == len(metadatas)

    client.batch.configure(batch_size=batch_size, dynamic=True)
    with client.batch as batch:
        for i in range(n):
            batch.add_data_object(
                data_object={
                    "text": texts[i],
                    "metadata": metadatas[i],
                },
                class_name=class_name,
                uuid=ids[i],
                vector=embeddings[i],
            )
