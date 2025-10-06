# weaviate_client.py — tương thích v4, có fallback v4 cũ
from typing import List
import weaviate
from weaviate.classes.config import Property, DataType, Configure


def _connect_local():
    """Kết nối Weaviate local (HTTP 8080, gRPC 50051)."""
    return weaviate.connect_to_local()


def _make_vector_config():
    """
    Trả về cấu hình vector 'bring-your-own vectors'.
    v4.16+  : Configure.Vectors.self_provided()
    v4 cũ   : Configure.Vectorizer.none()
    """
    if hasattr(Configure, "Vectors"):  # v4.16+
        return ("vector_config", Configure.Vectors.self_provided())
    # fallback client cũ
    return ("vectorizer_config", Configure.Vectorizer.none())


def ensure_schema(class_name: str):
    """
    Tạo collection tối giản cho BYOV:
      - vector: tự cung cấp
      - properties:
          text: TEXT
          metadata: OBJECT{ title: TEXT, abstract: TEXT, keywords: TEXT[], created_date: DATE }
    """
    client = _connect_local()
    try:
        if client.collections.exists(class_name):
            return

        key, vec_cfg = _make_vector_config()
        kwargs = {
            "name": class_name,
            key: vec_cfg,
            "properties": [
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
        }

        try:
            client.collections.create(**kwargs)
        except TypeError:
            # Thêm một lớp fallback cuối cho môi trường lai tạp
            client.collections.create(
                name=class_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=kwargs["properties"],
            )
    finally:
        client.close()


def upsert_chunks(class_name, ids, texts, embeddings, metadatas, batch_size: int = 64):
    import weaviate
    client = weaviate.connect_to_local()
    try:
        col = client.collections.get(class_name)

        # Cách 1: dynamic (tự điều chỉnh), KHÔNG truyền batch_size
        with col.batch.dynamic() as batch:
            for i in range(len(ids)):
                batch.add_object(
                    properties={"text": texts[i], "metadata": metadatas[i]},
                    uuid=ids[i],
                    vector=embeddings[i],
                )

        # Cách 2: cố định kích thước batch (nếu muốn):
        # with col.batch.fixed_size(batch_size=batch_size) as batch:
        #     for i in range(len(ids)):
        #         batch.add_object(
        #             properties={"text": texts[i], "metadata": metadatas[i]},
        #             uuid=ids[i],
        #             vector=embeddings[i],
        #         )
    finally:
        client.close()
