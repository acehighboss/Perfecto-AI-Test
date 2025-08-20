import re
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_KR_STOP = set(["그리고","그러나","하지만","또한","이는","이것은","그것은","수","등","것","때문","정도","에서","으로","에게","하다","했다","하는","하면"])

def _sent_tokenize_kr(text: str) -> List[str]:
    """
    줄바꿈 우선, 문장부호 보조. (외부 의존성 없이 안전)
    """
    parts = []
    for blk in re.split(r"\n+", text.strip()):
        blk = blk.strip()
        if not blk:
            continue
        # 문장부호 기준 보조 분리
        segs = re.split(r"(?<=[\.!\?…])\s+|(?<=다)\s+|(?<=요)\s+", blk)
        for s in segs:
            s = s.strip()
            if s:
                parts.append(s)
    return parts

def _tokenize(s: str) -> List[str]:
    s = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", s)
    toks = [t for t in s.lower().split() if t not in _KR_STOP and len(t) > 1]
    return toks

def _overlap_score(q_tokens: List[str], s_tokens: List[str]) -> float:
    if not s_tokens:
        return 0.0
    qs = set(q_tokens)
    ss = set(s_tokens)
    inter = len(qs & ss)
    # 길이가 너무 긴 문장 페널티(가벼운 정규화)
    return inter / (len(ss) ** 0.5 + 1e-6)

def _format_timecode(sec: float) -> str:
    try:
        sec = int(round(float(sec)))
    except Exception:
        return ""
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _build_citation(meta: Dict, fallback: str = "") -> str:
    """
    유튜브: 타임코드 / PDF: 페이지 / 그 외: URL 또는 파일명
    """
    src = meta.get("source") or meta.get("url") or meta.get("file_path") or fallback
    title = meta.get("title")
    kind = (meta.get("source_type") or meta.get("type") or "").lower()
    # PDF 페이지
    page = meta.get("page") or meta.get("page_number") or meta.get("pageIndex")
    # YouTube 시간
    start = meta.get("start") or meta.get("start_time") or meta.get("timestamp")

    # 유튜브 추정
    is_yt = "youtube" in str(src).lower() or kind == "youtube" or meta.get("is_youtube")
    # pdf 추정
    is_pdf = str(src).lower().endswith(".pdf") or kind == "pdf" or meta.get("is_pdf")

    if is_yt and start is not None:
        tc = _format_timecode(start)
        label = f"{title or 'YouTube'} @ {tc}" if tc else f"{title or 'YouTube'}"
        return f"{label} | {src}" if src else label

    if is_pdf and page is not None:
        label = f"{title or 'PDF'} p.{page}"
        return f"{label} | {src}" if src else label

    # 일반 웹/텍스트
    if title and src:
        return f"{title} | {src}"
    return src or title or "출처 미상"

def extract_relevant_sentences(query: str, docs: List[Document], max_sentences: int = 8) -> List[Dict]:
    """
    각 문서를 문장 단위로 쪼개 간단한 토큰 겹침 점수로 랭킹 → 상위 문장만 수집.
    반환: [{ "text": 문장, "score": 점수, "citation": 문자열, "meta": 메타 }, ... ]
    """
    q_tok = _tokenize(query)
    pool: List[Tuple[float, Dict]] = []
    for d in docs:
        text = d.page_content or ""
        meta = d.metadata or {}
        for sent in _sent_tokenize_kr(text):
            s_tok = _tokenize(sent)
            score = _overlap_score(q_tok, s_tok)
            if score <= 0:
                continue
            pool.append((score, {
                "text": sent,
                "meta": meta,
                "citation": _build_citation(meta),
                "score": score,
            }))

    # 점수 내림차순 → 출처 다양성 확보 위해 같은 소스 연속 과다 선택 방지
    pool.sort(key=lambda x: x[0], reverse=True)

    picked: List[Dict] = []
    seen_per_source = {}
    for _, item in pool:
        src_key = (item["meta"].get("source") or item["meta"].get("url") or "unknown")
        if seen_per_source.get(src_key, 0) >= 3:
            continue
        picked.append(item)
        seen_per_source[src_key] = seen_per_source.get(src_key, 0) + 1
        if len(picked) >= max_sentences:
            break

    # 문장이 1개도 없으면, 상위 문서의 앞부분을 안전하게 대체
    if not picked and docs:
        d0 = docs[0]
        meta = d0.metadata or {}
        fallback = (d0.page_content or "").strip().split("\n", 1)[0][:300]
        picked = [{
            "text": fallback,
            "meta": meta,
            "citation": _build_citation(meta),
            "score": 0.0
        }]
    return picked

def build_answer_from_sentences(llm, query: str, picked: List[Dict]) -> str:
    """
    반드시 인용문만 근거로 대답하도록 강제. 출력 끝에 [출처] 블록에 사용 문장과 출처를 나열.
    """
    numbered = []
    for i, it in enumerate(picked, 1):
        numbered.append(f"[S{i}] {it['text']}\n    └ 출처: {it['citation']}")
    context = "\n\n".join(numbered)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 신중한 RAG 비서다. 아래 [근거문장]에 포함된 내용만을 사용해 한국어로 답하라. "
         "모호하면 '제공된 출처로는 확정하기 어렵습니다'라고 말하라. "
         "추측하지 말고, 인용문 밖 내용을 가져오지 마라."),
        ("human",
         "질문: {query}\n\n[근거문장]\n{context}\n\n"
         "요청사항:\n- 위 문장들로만 답변을 구성\n- 과도한 요약 금지, 핵심만 명확히\n- 마지막에 [출처] 섹션으로 사용한 문장과 출처를 그대로 나열")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "context": context})
