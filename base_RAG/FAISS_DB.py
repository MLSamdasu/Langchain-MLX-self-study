"""
FAISS란?
Facebook AI Similarity Search
-> Meta에서 만든 고속 벡터 검색 라이브러리
-> ChromaDB와 같은 역할이지만, 대규모 데이터에서 훨씬 빠름
-> 저장방식 : 바이너리 인덱스 파일 (chromaDB -> SQLite + 파일)
-> 프로토타입에는 chromaDB, 속도 중요와 배포는 FAISS
"""

from annotated_types import doc
from langchain_community.vectorstores import FAISS

# FAISS 벡터 스토어 생성
faiss_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)
# 로컬 저장/ 로드
faiss_store.save_local("./faiss_index")
faiss_store = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,  # pickle 기반이라 이 옵션 필요
    # pickle -> 객체를 파일로 저장하고, 나중에 다시 불러오는 기능
    # pickle 파일을 불러올때, 파이썬은 파일 안에 담긴 명령을 그대로 실행함
    # 단순히 데이터만 읽는게 아니라서 보안 위험이 있음
    # Langchain은 기본값으로 로드를 막아둠 -> True로 해야 사용가능함
)

# Retriever 로 변환 (k=4: 상위 4개 청크)
retriever = faiss_store.as_retriever(search_kwargs={"k": 4})
