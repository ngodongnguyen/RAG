import json
import requests
from typing import List, Dict, Any, Literal
from openai import OpenAI

# ================== SYSTEM PROMPT ==================
META_SYS_PROMPT = (
    "Bạn là hệ thống trích xuất metadata.\n"
    "Hãy đọc đoạn văn bản đầu vào và tạo ra danh sách metadata cho từng mục sản phẩm riêng biệt.\n"
    "Mỗi metadata bao gồm 4 trường:\n"
    "- created_date: ngày hiện tại (YYYY-MM-DD)\n"
    "- title: tiêu đề sản phẩm (dòng chính)\n"
    "- abstract: mô tả ngắn gọn nội dung của sản phẩm (1–2 câu)\n"
    "- keywords: danh sách các từ khóa quan trọng, số liệu, điều kiện hoặc cụm ý chính.\n\n"
    "Yêu cầu:\n"
    "- Nếu văn bản chứa nhiều mục (vd: Vay cá nhân, Thẻ tín dụng, v.v.) thì mỗi mục là một object riêng.\n"
    "- Đầu ra phải là JSON array hợp lệ.\n\n"
    "Ví dụ:\n"
    "Đầu vào:\n"
    "Tổng quan sản phẩm\n"
    "Vay cá nhân FEC\n"
    "- Chỉ cần CCCD\n"
    "- Hạn mức vay tối đa 100 triệu\n\n"
    "Fast Card\n"
    "- Thẻ tín dụng dành cho DN vừa và nhỏ (SME)\n\n"
    "Đầu ra:\n"
    "[\n"
    "  {\n"
    '    \"created_date\": \"2025-10-05\",\n'
    '    \"title\": \"Vay cá nhân FEC\",\n'
    '    \"abstract\": \"Sản phẩm vay tiền mặt cho khách hàng cá nhân, chỉ cần CCCD, hạn mức tối đa 100 triệu đồng.\",\n'
    '    \"keywords\": [\"vay cá nhân\", \"FEC\", \"CCCD\", \"hạn mức 100 triệu\", \"vay tiêu dùng\"]\n'
    "  },\n"
    "  {\n"
    '    \"created_date\": \"2025-10-05\",\n'
    '    \"title\": \"Fast Card\",\n'
    '    \"abstract\": \"Thẻ tín dụng dành cho doanh nghiệp vừa và nhỏ (SME).\",\n'
    '    \"keywords\": [\"Fast Card\", \"thẻ tín dụng\", \"doanh nghiệp SME\", \"FEC\", \"tài chính doanh nghiệp\"]\n'
    "  }\n"
    "]\n\n"
    "Hãy trích xuất metadata cho văn bản sau:"
)

# ================== CORE FUNCTION ==================
client = OpenAI(api_key="YOUR_API_KEY_HERE")

def extract_metadata(
    chunk: str,
    backend: Literal["ollama", "gpt"] = "ollama",
    model_ollama: str = "llama3.2:3b",
    model_gpt: str = "gpt-4o-mini",
    endpoint_ollama: str = "http://10.1.1.237:11434",
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Hàm chính — có thể switch backend giữa Ollama hoặc GPT-4o-mini.
    """
    try:
        if backend == "ollama":
            # --- Ollama ---
            prompt = f"{META_SYS_PROMPT}\n\nVăn bản:\n'''{chunk}'''"
            r = requests.post(
                f"{endpoint_ollama}/api/generate",
                json={"model": model_ollama, "prompt": prompt, "stream": False},
                timeout=timeout
            )
            r.raise_for_status()
            txt = r.json().get("response", "").strip()

        elif backend == "gpt":
            # --- GPT-4.0-mini ---
            res = client.chat.completions.create(
                model=model_gpt,
                messages=[
                    {"role": "system", "content": META_SYS_PROMPT},
                    {"role": "user", "content": chunk}
                ],
                temperature=0,
                timeout=timeout,
            )
            txt = res.choices[0].message.content.strip()

        else:
            raise ValueError("backend phải là 'ollama' hoặc 'gpt'.")

        return json.loads(txt)

    except Exception as e:
        return {
            "error": str(e),
            "created_date": None,
            "title": None,
            "abstract": None,
            "keywords": []
        }


def extract_metadata_batch(
    chunks: List[str],
    backend: Literal["ollama", "gpt"] = "ollama"
) -> List[Dict[str, Any]]:
    """
    Chạy batch cho nhiều đoạn văn bản, hỗ trợ switch backend.
    """
    return [extract_metadata(ch, backend=backend) for ch in chunks]


