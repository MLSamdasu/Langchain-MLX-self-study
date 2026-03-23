# 임베딩 된 벡터를 저장하는곳 -> 벡터DB -> 그중 ChromaDB
# 파이썬 웹 개발이라고 검색하면 의미적으로 가까운 Django 튜토리얼, Flask로 만드는 RestAPI와 같은 의미적으로 유사한 문서를 함께 찾아줌
"""
ChromaDB
SQLite 기반 동작 -> 가벼움
별도의 서버 없이 프로세스 내에서 실행 -> Redis나 PostgreSQL처럼 서버를 따로 띄울 필요 없음
pip install로 바로 설치 가능함

처음 생성할때는 Chroma.from_documents(),
이미 저장된 것을 불러올때는 Chroma()

주의 -> 로드시 매개변수 이름이 embedding -> embedding_function으로 바뀜
"""

from langchain_chroma import Chroma

# 문서 청크로 벡터 DB 생성
Vectorstore = Chroma.from_documents(
    documents=chunks,  # 저장할 텍스트 조각들
    embedding=embeddings,  # 임베딩 모델(multilingual-e5-large) 텍스트 -> 벡터 변환
    persist_directory="./chroma_db",  # 디스크에 저장할 경로 -> 저장 폴더 경로
    collection_name="my_docs",  # 컬렉션 이름(DB안의 테이블 이름과 같은것)
)

# 이후 세션에서 로드 (from_documents 없이)
Vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_docs",
)

# 유사도 검색 테스트
results = Vectorstore.similarity_search(
    "Langchain 체인 구성 방법",  # 검색 쿼리
    k=3,  # 상위 3개의 결과 반환
)
"""
similarity_search() 동작 과정
1. 검색 쿼리를 임베딩 모델로 벡터 변환
2. DB에 저장된 모든 벡터와 코사인 유사도 비교
3. 가장 유사한 상위 k개 문서를 변환 
"""

for doc in results:
    print(doc.page_content[:100])
