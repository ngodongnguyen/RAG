from typing import List, Dict, Any
import hashlib
from sentence_transformers import SentenceTransformer

def embed_chunks_st(chunks: List[str],
                    model_name: str = "BAAI/bge-m3",
                    batch: int = 64,
                    device: str = None) -> List[Dict[str, Any]]:
    mdl = SentenceTransformer(model_name, device=device)
    out: List[Dict[str, Any]] = []
    for i in range(0, len(chunks), batch):
        part = chunks[i:i+batch]
        ids = [hashlib.md5(c.encode("utf-8")).hexdigest() for c in part]
        embs = mdl.encode(part, batch_size=batch, convert_to_numpy=True,
                          normalize_embeddings=False, show_progress_bar=False)
        for cid, vec in zip(ids, embs):
            out.append({"id": cid, "embedding": vec.tolist(), "n_dims": len(vec)})
    return out
