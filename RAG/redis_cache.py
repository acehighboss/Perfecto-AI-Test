import hashlib
import json
from typing import Any

def _normalize_for_hash(content: Any) -> bytes:
    """캐시 해시 입력을 bytes로 표준화."""
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")

    # dict/list/tuple 등: 정렬된 JSON으로 직렬화
    try:
        normalized = json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            default=str  # 직렬화 불가 객체 방지
        )
        return normalized.encode("utf-8")
    except Exception:
        # 최후의 수단
        return str(content).encode("utf-8")

def create_cache_key(prefix: str, content: Any) -> str:
    """prefix:md5 해시 형태의 캐시 키 생성 (타입 안전)."""
    body = _normalize_for_hash(content)
    digest = hashlib.md5(body).hexdigest()
    return f"{prefix}:{digest}"
