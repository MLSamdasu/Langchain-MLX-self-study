# RAG 검색을 Agent의 Tool로 등록하면, Agent가 필요할 때만 문서를 검색한다.
# 일반 RAG체인은 모든 질문에 무조건 벡터 검색을 함
# Agent + RAG 조합에서는 LLM이 이 질문에 검색이 필요한지를 먼저 판단함
# 검색이 필요없는 일반 질문에는 직접 답변하므로 효율적임
"""
Agent-RAG 결합 패턴 (Flow)

# 일반 RAG: 항상 검색 실행
question → retriever → LLM → answer

# Agent + RAG: 필요할 때만 검색
question → Agent(판단) → search_docs() or 직접 답변
"""


from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 내 폴더에서 만든 파일명으로 import
from Tools_Decorator import calculate, search_web
from StructuredTool_Pydantic import file_reader

# 벡터 DB 로드
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large", model_kwargs={"device": "mps"}
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


# docs string 작성 팁
# 1. "~할때 사용하세요"로 사용 조건 명시
# 2. 다른 Tool과 경계가 명확하도록 작성 (search_docs vs search_web 차이를 LLM이 구분할 수 있도록 해야함)
# 3. 입력 파라미터가 어떤 형태여야 하는지 예시 포함


@tool
def search_docs(query: str) -> str:
    # docs의 string이 품질임 -> 이걸 보고 호출여부를 결정하기 떄문.
    # 대충쓰면 특히 9B모델 수준에서는 tool 선택 오류가 급증함
    """
    내부 문서에서 정보를 검색합니다.
    기술문서, 매뉴얼 관련 질문에 사용하세요.
    """
    docs = vectorstore.similarity_search(query, k=3)
    return "\n---\n".join(d.page_content for d in docs)


# Agent에 RAG tool 추가
tools = [calculate, search_docs, search_web]
agent = create_agent(model=llm, tools=tools)
