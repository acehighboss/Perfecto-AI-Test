import re
import math
from typing import List, Tuple, Dict, Any
from collections import Counter
from langchain_core.documents import Document

# ---------- 문장 분할 ----------
def _split_sentences(text: str) -> List[str]:
    """kss가 있으면 사용, 없으면 정규식 사용."""
    text = text or ""
    try:
        import kss  # optional
        return [s.strip() for s in kss.split_sentences(text) if s.strip()]
    except Exception:
        # 마침표/물음표/느낌표 및 줄바꿈 기준
        parts = re.split(r"(?<=[\.!?])\s+|\n+", text)
        return [p.strip() for p in parts if p and p.strip()]

# ---------- 토큰화 ----------
_WORD_RE = re.compile(r"[A-Za-z0-9]+|[가-힣]+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]

# ---------- TF-IDF 기반 간단 문장 랭킹 ----------
def _rank_sentences_by_query(query: str, sentences: List[str]) -> List[Tuple[int, float]]:
    """
    간단 TF-IDF: 질문 토큰과의 교집합 토큰에 대해서만 가중치 합산.
    반환: [(문장인덱스, 점수), ...] 내림차순
    """
    if not sentences:
        return []

    q_tokens = set(_tokenize(query))
    sent_tokens = [_tokenize(s) for s in sentences]

    # DF/IDF 계산(문장 단위)
    df = Counter()
    for toks in sent_tokens:
        uniq = set(toks)
        for t in uniq:
            df[t] += 1

    N = len(sentences)
    def idf(t: str) -> float:
        return math.log(1.0 + N / (1.0 + df.get(t, 0)))

    scores: List[Tuple[int, float]] = []
    for idx, toks in enumerate(sent_tokens):
        if not toks:
            scores.append((idx, 0.0))
            continue
        tf = Counter(toks)
        # 질문과 겹치는 토큰들만 반영
        common = q_tokens.intersection(tf.keys())
        s = sum(tf[t] * idf(t) for t in common)
        scores.append((idx, s))

    # 점수 내림차순
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# ---------- 메타데이터에서 페이지/타임코드 추출 ----------
def _find_page(meta: Dict[str, Any]) -> Any:
    if meta is None:
        return None
    if "page" in meta:
        return meta["page"]
    if "page_number" in meta:
        return meta["page_number"]
    loc = meta.get("loc")
    if isinstance(loc, dict) and "page" in loc:
        return loc["page"]
    return None

def _find_timecode(meta: Dict[str, Any]) -> str | None:
    # 초 단위 start/end or 문자열 타임코드가 들어오는 케이스를 모두 흡수
    for key in ("start", "start_time", "timestamp", "yt_timecode"):
        if key in meta and meta[key]:
            if isinstance(meta[key], (int, float)):
                secs = int(meta[key])
                m, s = divmod(secs, 60)
                return f"{m:02d}:{s:02d}"
            return str(meta[key])
    return None

def _format_source_label(meta: Dict[str, Any]) -> str:
    src = meta.get("source") or meta.get("url") or meta.get("file_path") or "source"
    page = _find_page(meta)
    tc = _find_timecode(meta)
    suffix = []
    if page is not None:
        suffix.append(f"p.{page}")
    if tc:
        suffix.append(f"t={tc}")
    return f"{src}" + (f" ({', '.join(suffix)})" if suffix else "")

# ---------- 문서 -> 최소 문장들로 축약 ----------
def _prune_docs_to_min_sentences(
    question: str,
    docs: List[Document],
    per_doc_max: int = 3
) -> List[Document]:
    """각 문서에서 질문과 가장 관련 높은 문장들만 남겨 Document.page_content를 축약.
    또한 meta['selected_sentences']에 [{text, label}] 저장."""
    pruned: List[Document] = []
    for d in docs:
        sents = _split_sentences(d.page_content or "")
        if not sents:
            pruned.append(d)
            continue

        ranked = _rank_sentences_by_query(question, sents)
        keep_idxs = [i for i, score in ranked[:max(1, per_doc_max)] if score > 0]
        # 점수가 모두 0이라면 첫 문장만
        if not keep_idxs and ranked:
            keep_idxs = [ranked[0][0]]

        kept = [sents[i] for i in sorted(keep_idxs)]
        # 문장별 라벨(출처 + p./t=)
        label = _format_source_label(d.metadata or {})
        citations = [{"text": s, "label": label} for s in kept]

        new_meta = dict(d.metadata or {})
        new_meta["selected_sentences"] = citations

        pruned.append(
            Document(
                page_content=" ".join(kept),
                metadata=new_meta
            )
        )
    return pruned

# ====== 기존 retrieve 함수 대체 ======
# 기존에 있던 retrieve 함수 이름이 아래와 동일하다면 이 버전으로 교체하세요.
def retrieve_and_fuse_results(retriever, queries: List[str]) -> List[Document]:
    """
    기존: retriever.batch(queries) -> 문서 리스트 반환
    변경: 동일 반환이지만, 각 Document.page_content를 질문과 관련 높은 문장만 남기도록 축약하고
         metadata['selected_sentences']에 문장별 출처 라벨을 담아둠.
    """
    retrieved_docs_lists = retriever.batch(queries)
    # 평탄화
    flat_docs: List[Document] = []
    for lst in retrieved_docs_lists:
        if lst:
            flat_docs.extend(lst)

    question = queries[0] if queries else ""
    pruned = _prune_docs_to_min_sentences(question, flat_docs, per_doc_max=3)
    return pruned
