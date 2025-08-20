# RAG/redis_cache.py
# 타입 안전한 캐시 키 + Redis(있으면) / 메모리(없으면) 캐시
import os
import time
import json
import hashlib
import threading
from typing import Any, Optional

try:
    import redis  # optional
except Exception:
    redis = None  # redis 미설치/미사용 환경 대비

# ---------------------------
# Redis 클라이언트 초기화
# ---------------------------
_CLIENT = None
_LOCK = threading.Lock()
_MEMORY_CACHE: dict[str, tuple[str, Optional[float]]] = {}  # key -> (data(str), expires_at)

def _get_client():
    """환경변수 기반으로 Redis 클라이언트를 생성 (없으면 None)."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    url = os.getenv("REDIS_URL")
    host = os.getenv("REDIS_HOST")
    if not redis:
        _CLIENT = None
        return _CLIENT

    try:
        if url:
            _CLIENT = redis.StrictRedis.from_url(url, decode_responses=True)
        else:
            _CLIENT = redis.StrictRedis(
                host=host or "127.0.0.1",
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
            )
    except Exception:
        _CLIENT = None
    return _CLIENT

# ---------------------------
# 해시 입력 표준화
# ---------------------------
def _normalize_for_hash(content: Any) -> bytes:
    """캐시 해시 입력을 bytes로 표준화."""
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    try:
        normalized = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
        return normalized.encode("utf-8")
    except Exception:
        return str(content).encode("utf-8")

def create_cache_key(prefix: str, content: Any) -> str:
    """prefix:md5 형태의 캐시 키 생성 (타입 안전)."""
    body = _normalize_for_hash(content)
    digest = hashlib.md5(body).hexdigest()
    return f"{prefix}:{digest}"

# ---------------------------
# 직렬화/역직렬화
# ---------------------------
def _serialize(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "ignore")
        except Exception:
            return value.hex()
    if isinstance(value, str):
        return value
    # dict/list/tuple 등은 JSON으로
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)

def _deserialize(s: Optional[str]) -> Any:
    if s is None:
        return None
    s2 = s.strip()
    if not s2:
        return None
    # JSON 시도 -> 실패 시 원문 반환
    try:
        return json.loads(s2)
    except Exception:
        return s

# ---------------------------
# 캐시 API (retriever_builder가 기대)
# ---------------------------
def set_to_cache(key: str, value: Any, ttl_seconds: Optional[int] = 3600) -> None:
    """key에 value를 저장. Redis가 없으면 메모리 캐시 사용."""
    data = _serialize(value)
    client = _get_client()
    if client:
        try:
            if ttl_seconds and ttl_seconds > 0:
                client.setex(key, ttl_seconds, data)
            else:
                client.set(key, data)
            return
        except Exception:
            pass  # Redis 실패 시 메모리 fallback

    # 메모리 캐시
    with _LOCK:
        expires_at = time.time() + ttl_seconds if ttl_seconds and ttl_seconds > 0 else None
        _MEMORY_CACHE[key] = (data, expires_at)

def get_from_cache(key: str) -> Any:
    """key에 해당하는 값을 반환. 없거나 만료되면 None."""
    client = _get_client()
    if client:
        try:
            data = client.get(key)
            return _deserialize(data)
        except Exception:
            pass  # Redis 실패 시 메모리 fallback

    with _LOCK:
        item = _MEMORY_CACHE.get(key)
        if not item:
            return None
        data, expires_at = item
        if expires_at and time.time() > expires_at:
            # 만료
            try:
                del _MEMORY_CACHE[key]
            except Exception:
                _MEMORY_CACHE.pop(key, None)
            return None
    return _deserialize(data)
