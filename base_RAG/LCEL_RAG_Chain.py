"""
RAG 내용 총 정리 + Langchain에 연결

1. RunnableParallel
사용자가 "LCEL이란 무엇인가"라는 프롬프트 질문을 하면, 두가지 정보가 동시에 필요함
    1. context : 벡터DB에서 검색한 관련 문서들
    2. question : 사용자의 원래 질문 그 자체
    ->
    RunnableParallel(
        context = retriever | format_docs, <- 경로 A -> 검색 후 포맷
        question = RunnablePassthrough()  <- 경로 B -> 질문을 그대로 전달
    )

2. format_docs 함수가 왜 필요한가
retriever가 반환하는것 -> Document 객체 리스트 -> 프롬프트의 {context}에는 문자열이 들어가야함
->
def format_docs(docs):
    return "\n\n ---- \n\n".join(
        f"[출처: {d.metadata.get('source', '?')}]\n"
        f"{d.page_content}"
        for d in docs
    )
---------------------- 정리 ---------------------------
  데이터 흐름 추적: "LCEL이란 무엇인가요?"

  입력: "LCEL이란 무엇인가요?"
           │
      ┌────┴────┐
      ▼         ▼
   [경로 A]   [경로 B]         ← RunnableParallel (동시 실행)
   retriever  RunnablePassthrough
      │              │
      ▼              ▼
   [Doc1~4]   "LCEL이란 무엇인가요?"
      │
      ▼
   format_docs
      │
      ▼
   "[출처: doc1]\nLCEL은..."
      │              │
      └────┬─────────┘
           ▼
   {"context": "...", "question": "LCEL이란 무엇인가요?"}
           │
           ▼
      rag_prompt  → {context}와 {question}에 값 주입
           │
           ▼
      llm (MLX)  → 답변 생성
           │
           ▼
      StrOutputParser()  → 텍스트만 추출
           │
           ▼
      "LCEL은 LangChain Expression Language의..."

이전 내용 -> 프롬프트 | LLM | 파서
이번 RAG에서 배운것 -> RunnableParallel | 프롬프트 | LLM | 파서

"""

from torchgen.utils import context
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# MLX 로컬 모델
llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="X",
    model="mlx-community/Qwen3.5-9B-4bit",
)

# Retriever -> 백터저장소를 검색기로 변환함
# 변환 이유 : LCEL(체인)과의 호환성 -> | 를 사용하려면 Runnable 객체여야함
# 중복 제외와 같은 복잡한 검색 설정을 미리 담아둘 수 있음
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# RAG프롬프트
rag_prompt = ChatPromptTemplate.from_template(
    """
    다음 컨텍스트를 참고하여 질문에 답하여라.
    컨텍스트에 없는 정보는 모른다고 답하라.
    컨텍스트 : {context}
    질문 : {question}
    답변 : 
    """
)


# 컨텍스트 포맷팅 함수
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[출처: {d.metadata.get('source','?')}]\n" f"{d.page_content}" for d in docs
    )


# 완전한 RAG 체인
rag_chain = RunnableParallel(
    context=retriever | format_docs, question=RunnablePassthrough()
)
