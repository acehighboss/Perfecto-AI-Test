from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass
import re
from collections import Counter, defaultdict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# (옵션) BM25
try:
    from langchain.retrievers import BM25Retriever
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

# (옵션) 리랭커 우선순위: FlagEmbedding -> sentence-transformers -> 없음
def _try_load_reranker():
    try:
        from FlagEmbedding import FlagReranker
        return ("flag", FlagReranker("BAAI/bge-reranker-large", use_fp16=True))
    except Exception:
        try:
            from sentence_transformers import CrossEncoder
            return ("ce", CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"))
        except Exception:
            return (None, None)

# 간단 언어 감지(한글 존재 여부)
HAN_RE = re.compile(r"[가-힣]")
WORD_RE = re.compile(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣_\-+/]{1,}")

KO_STOP = {
    "그리고","그러나","하지만","또한","및","또","이","그","저","것","수","등","때문","대한",
    "에서","으로","에게","에서의","에는","에는요","에는요?","에","도","만","로","다","요","가","은","는","을","를",
    "하다","되어","된다","있는","하는","됐다","했다","있다","없다","같은","에는"
}
EN_STOP = {
    "the","a","an","and","or","but","if","then","else","to","for","of","in","on","at","by","with",
    "this","that","these","those","is","are","was","were","be","been","being","it","as","from","into",
    "we","you","they","i","he","she","them","his","her","our","their","your","not"
}

def _lang_stopwords(text: str):
    return (KO_STOP if HAN_RE.search(text or "") else EN_STOP)

def _tokenize(text: str, lang_stop: set) -> List[str]:
    if not text:
        return []
    toks = [t.lower() for t in WORD_RE.findall(text)]
    return [t for t in toks if t not in lang_stop and len(t) >= 2]

def _extract_dynamic_hints(query: str, seed_docs: List[Document], topk_doc_tokens:int=10) -> List[str]:
    """
    질의 + 초기 검색 상위 문서에서 동적으로 핵심 토큰을 뽑아 '도메인 힌트'로 사용.
    - 질의 토큰
    - 초기 문서들에서 자주 등장하는 토큰(여러 문서에 중복 등장하는 것 우선)
    """
    lang_stop = _lang_stopwords(query)
    q_tokens = _tokenize(query, lang_stop)

    # 문서 토큰 빈도(문서 빈도와 전체 빈도 혼합)
    df = Counter()  # in how many docs the term appears
    tf = Counter()  # total frequency across docs
    for d in seed_docs:
        toks = set(_tokenize(d.page_content or "", lang_stop))
        for t in toks:
            df[t] += 1
        tf.update(_tokenize(d.page_content or "", lang_stop))

    # 문서 빈도가 높은 토큰 우선 -> 동률이면 TF 큰 것 우선
    # 너무 일반적인 숫자/연도 토큰은 가볍게 필터
    def _bad(t: str) -> bool:
        return t.isdigit() and (len(t) <= 4)  # 2025 같은 연도토큰 남발 방지

    cand = [t for t,_ in df.most_common() if not _bad(t)]
    cand.sort(key=lambda t: (df[t], tf[t]), reverse=True)

    # 힌트 후보 구성: 질의 토큰 + 문서 토큰 상위
    hints = list(dict.fromkeys(q_tokens + cand[:topk_doc_tokens]))
    return hints

def _has_signals(docs: List[Document], hints: List[str], min_hits: int = 1) -> bool:
    if not docs or not hints:
        return False
    set_hints = set(hints)
    hits = 0
    for d in docs:
        txt = (d.page_content or "").lower()
        if any(h in txt for h in set_hints):
            hits += 1
            if hits >= min_hits:
                return True
    return False

def _rerank(query: str, docs: List[Document], top_n: int = 4) -> List[Document]:
    kind, model = _try_load_reranker()
    if not docs:
        return []
    if kind is None:
        # 키워드 기반 간이 스코어(의존성 없을 때)
        lang_stop = _lang_stopwords(query)
        q_tokens = set(_tokenize(query, lang_stop))
        scored = []
        for d in docs:
            t = set(_tokenize(d.page_content or "", lang_stop))
            inter = len(q_tokens & t)
            scored.append((inter, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_n]]

    # 크로스 인코더/Flag 리랭커
    pairs = [[query, d.page_content or ""] for d in docs]
    try:
        if kind == "flag":
            scores = model.compute_score(pairs)
        else:
            scores = model.predict(pairs)
        ranked = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        return ranked[:top_n]
    except Exception:
        return docs[:top_n]

@dataclass
class SmartRetrieverConfig:
    top_n: int = 4
    escalate_k: Tuple[int, ...] = (6, 10, 16)
    use_bm25: bool = True
    use_reranker: bool = True
    min_domain_hits: int = 1          # '힌트가 들어있는 문서' 최소 개수
    debug: bool = False               # 디버그 로깅

class SmartRetriever(BaseRetriever):
    """
    search_type='similarity' 기반 base_retriever를 감싸 범용적으로 보강.
    1) similarity로 점증적 k 탐색
    2) 질의+초기문서에서 '동적 힌트' 추출
    3) 힌트가 부족하면 BM25로 보강 (가능할 때)
    4) (가능하면) 리랭킹
    5) 비어 있으면 완충
    """
    def __init__(
        self,
        base_retriever: BaseRetriever,
        all_docs: Optional[List[Document]] = None,
        config: SmartRetrieverConfig = SmartRetrieverConfig(),
        logger=None,  # streamlit st or print 대체용
    ):
        self.base_retriever = base_retriever
        self.all_docs = all_docs or []
        self.cfg = config
        self._logger = logger

    def _log(self, *args):
        if not self.cfg.debug:
            return
        try:
            if self._logger is not None and hasattr(self._logger, "write"):
                self._logger.write(*args)
            elif self._logger is not None and hasattr(self._logger, "markdown"):
                self._logger.markdown(" ".join(str(a) for a in args))
            else:
                print(*args)
        except Exception:
            pass

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1) similarity 단계: k를 키워가며 검색
        first = []
        used_k = None
        for k in self.cfg.escalate_k:
            if hasattr(self.base_retriever, "search_kwargs"):
                self.base_retriever.search_kwargs["k"] = k
            cand = self.base_retriever.get_relevant_documents(query)
            first = cand
            used_k = k
            if cand:
                hints = _extract_dynamic_hints(query, cand, topk_doc_tokens=10)
                if _has_signals(cand, hints, min_hits=self.cfg.min_domain_hits):
                    break

        self._log(f"[SmartRetriever] similarity k={used_k}, first={len(first)}")

        candidates = list(first)

        # 2) 동적 힌트 준비
        hints = _extract_dynamic_hints(query, candidates, topk_doc_tokens=10)

        # 3) BM25 보강 (필요할 때만)
        if (
            self.cfg.use_bm25 and HAS_BM25 and
            not _has_signals(candidates, hints, min_hits=self.cfg.min_domain_hits) and
            len(self.all_docs) > 0
        ):
            bm25 = BM25Retriever.from_documents(self.all_docs)
            bm25.k = max(self.cfg.top_n * 2, 8)
            # 질의 확장(힌트 상위 몇 개를 덧붙임)
            expand = " ".join(hints[:8]) if hints else ""
            q2 = (query + " " + expand).strip()
            bm_hits = bm25.get_relevant_documents(q2)

            # 중복 제거 병합(문서+페이지+첫64자 기준)
            seen = set()
            def key(d: Document):
                m = d.metadata or {}
                return m.get("source","") + f"::{m.get('page','')}" + "::" + (d.page_content or "")[:64]
            merged = []
            for d in candidates + bm_hits:
                k = key(d)
                if k not in seen:
                    seen.add(k)
                    merged.append(d)
            candidates = merged
            self._log(f"[SmartRetriever] BM25 added: {len(bm_hits)} -> merged={len(candidates)}")

        # 4) 리랭커 (가능하면)
        if self.cfg.use_reranker:
            final = _rerank(query, candidates, top_n=self.cfg.top_n)
        else:
            final = candidates[: self.cfg.top_n]

        # 5) 최종 방어(비어 있지 않도록)
        if not final and candidates:
            final = candidates[: self.cfg.top_n or 4]

        # 6) 디버그: 선택된 힌트/소스 표시
        if self.cfg.debug:
            srcs = [ ( (d.metadata or {}).get("source",""), (d.metadata or {}).get("page","") ) for d in final ]
            self._log(f"[SmartRetriever] hints={hints[:10]}")
            self._log(f"[SmartRetriever] finals={len(final)} srcs={srcs}")

        return final
