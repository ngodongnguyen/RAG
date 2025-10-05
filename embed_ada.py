from typing import List, Dict, Any
import hashlib, time
from openai import OpenAI

def embed_chunks_ada(chunks: List[str],
                     model: str = "text-embedding-3-small",
                     batch: int = 64,
                     sleep: float = 0.1) -> List[Dict[str, Any]]:
    client = OpenAI()
    out = []
    for i in range(0, len(chunks), batch):
        part = chunks[i:i+batch]
        ids = [hashlib.md5(c.encode("utf-8")).hexdigest() for c in part]
        resp = client.embeddings.create(model=model, input=part)
        for cid, r in zip(ids, resp.data):
            vec = r.embedding
            out.append({"id": cid, "embedding": vec, "n_dims": len(vec)})
        if sleep: time.sleep(sleep)
    return out
