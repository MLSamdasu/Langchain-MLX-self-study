"""
4.9 출처 추적이 포함된 RAG

기존 4.8의 rag_chain(답변만 반환)을 수정하지 않고,
RunnableParallel로 한 번 더 감싸서 출처 정보를 함께 반환한다.

[데이터 흐름]

  "LCEL이란?"
       │
  RunnableParallel (바깥 — 4.9에서 추가)
       │
       ├── answer ── rag_chain ──────────→ "LCEL은..." (문자열)
       │                │
       │           RunnableParallel (안쪽 — 4.8 기존)
       │                ├── context: retriever | format_docs
       │                └── question: RunnablePassthrough()
       │                     │
       │                     ▼
       │               rag_prompt | llm | StrOutputParser()
       │
       └── sources ── retriever ─────────→ [Doc1, Doc2, ...] (Document 리스트)
       │
       ▼
  {"answer": "LCEL은...", "sources": [Doc1, Doc2, ...]}

"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# ============================================================
# 여기서부터 4.8과 동일 (기존 코드)
# ============================================================

# MLX 로컬 모델
llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="mlx-community/Qwen3.5-9B-4bit",
)

# Retriever — 벡터저장소를 검색기로 변환
# vectorstore는 ChromaDB.py 또는 FAISS_DB.py에서 생성한 것을 사용
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# RAG 프롬프트
rag_prompt = ChatPromptTemplate.from_template(
    """다음 컨텍스트를 참고하여 질문에 답하여라.
컨텍스트에 없는 정보는 모른다고 답하라.

컨텍스트: {context}

질문: {question}
답변:"""
)


# 컨텍스트 포맷팅 함수 — Document 리스트를 하나의 문자열로 합침
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[출처: {d.metadata.get('source', '?')}]\n{d.page_content}"
        for d in docs
    )


# 완전한 RAG 체인 (4.8)
# 반환값: str (답변 문자열만)
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# 여기서부터 4.9 추가 코드
# ============================================================

# 출처 포함 RAG 체인
# 기존 rag_chain을 수정하지 않고, 바깥에서 RunnableParallel로 감쌈
# 반환값: dict {"answer": str, "sources": list[Document]}
rag_with_sources = RunnableParallel(
    answer=rag_chain,       # 기존 체인 → 답변 생성
    sources=retriever,      # 같은 질문으로 검색 → 원본 Document 확보
)

# ============================================================
# 실행 및 출처 출력
# ============================================================

result = rag_with_sources.invoke("LCEL이란 무엇인가요?")

# 답변 출력
print("== 답변 ==")
print(result["answer"])

# 출처 출력 — Document의 metadata에서 source, page 추출
print("\n== 출처 ==")
for doc in result["sources"]:
    src = doc.metadata.get("source", "unknown")   # 파일 경로
    page = doc.metadata.get("page", "?")           # 페이지 번호
    print(f"  - {src} (p.{page})")
