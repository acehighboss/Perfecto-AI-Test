import asyncio
import spacy
from langchain_core.documents import Document as LangChainDocument
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_experimental.text_splitter import SemanticChunker

from .rag_config import RAGConfig
from .redis_cache import get_from_cache, set_to_cache, create_cache_key

# spaCy 언어 모델 로드 (앱 실행 시 한 번만 로드)
try:
    nlp_korean = spacy.load("ko_core_news_sm")
    nlp_english = spacy.load("en_core_web_sm")
    print("✅ spaCy language models loaded successfully.")
except OSError:
    print("⚠️ spaCy 모델을 찾을 수 없습니다. 'requirements.txt'에 모델이 포함되었는지 확인하세요.")
    nlp_korean, nlp_english = None, None


async def _sentence_split_and_embed_async(query: str, compression_retriever_1, embeddings):
    """(비동기) 1, 2단계 필터링 및 문장 분할, 임베딩, 최종 Rerank를 수행 (Redis, spaCy 적용)."""

    cache_key = create_cache_key("rag_result", query)
    cached_docs = get_from_cache(cache_key)
    if cached_docs is not None:
        return cached_docs

    print(f"\n[Cache Miss] 사용자 질문으로 1/2단계 필터링 실행: {query}")
    reranked_chunks = compression_retriever_1.invoke(query)
    print(f"1차 Rerank 후 {len(reranked_chunks)}개 청크 선별 완료.")

    sentences = []

    # ▼▼▼ [수정] 영어/한국어 처리 로직 모두에서 제너레이터를 리스트로 변환하여 오류 최종 해결 ▼▼▼
    if nlp_korean and nlp_english:
        for chunk in reranked_chunks:
            # page_content의 길이가 0인 경우 건너뛰기
            if not chunk.page_content or not chunk.page_content.strip():
                continue

            doc_ko = nlp_korean(chunk.page_content)
            sents_ko = list(doc_ko.sents)  # 한국어 제너레이터를 리스트로 변환

            # 휴리스틱: 한국어 문장이 1개 이하이고, 텍스트의 50% 이상이 알파벳이면 영어 모델로 재시도
            if len(sents_ko) <= 1 and sum(c.isalpha() and 'a' <= c.lower() <= 'z' for c in chunk.page_content) / len(chunk.page_content) > 0.5:
                doc_en = nlp_english(chunk.page_content)
                sents = [sent.text.strip() for sent in doc_en.sents]
            else:
                sents = [sent.text.strip() for sent in sents_ko]

            for i, sent_text in enumerate(sents):
                if sent_text:
                    metadata = chunk.metadata.copy()
                    metadata["chunk_location"] = f"chunk_{i+1}"
                    sentences.append(LangChainDocument(page_content=sent_text, metadata=metadata))
    else:
        print("spaCy 모델이 없어 문장 분할을 건너뜁니다. 청크를 그대로 사용합니다.")
        sentences = reranked_chunks
    # ▲▲▲ [수정] 여기까지 ▲▲▲

    print(f"총 {len(sentences)}개의 문장으로 분할 완료.")
    if not sentences:
        return []

    print("문장 임베딩 및 FAISS 인덱싱 시작...")
    sentence_texts = [s.page_content for s in sentences]

    async def embed_in_batches():
        all_embeddings = []
        for i in range(0, len(sentence_texts), RAGConfig.EMBEDDING_BATCH_SIZE):
            batch = sentence_texts[i:i + RAGConfig.EMBEDDING_BATCH_SIZE]
            print(f"임베딩 배치 {i // RAGConfig.EMBEDDING_BATCH_SIZE + 1} 처리 중 ({len(batch)}개 문장)...")
            batch_embeddings = await embeddings.aembed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    embedded_vectors = await embed_in_batches()
    
    text_embedding_pairs =
