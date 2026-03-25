# MMR (Maximal Marginal Relevance) -> 다양성 확보
# 이전에 했던것 -> search_kwargs로 유사도 높은 순서대로 그냥 뽑았음
# 문제 -> 1. 내용 중복 2. 노이즈(없어도 억지로 가까운것 가져옴)
# 해결 -> 1. MMR -> 다양성 확보, 2. Score Threshold -> 노이즈 차단, 3. MultiQueryRetriever -> 질문 자체를 풍부하게


# 1. MMR -> 관련성도 높으면서 서로 다른 내용을 가져오는 전략
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,  # 최종 반환할 문서의 수
        "fetch_k": 20,  # 후보 20개중 다양한 4개 선택
        "lambda_mult": 0.7,  # 0 = 최대 다양, 1 = 최대 유사
    },
)

# 2. Score Treshold -> 유사도 문턱값 설정
# 유사도가 0.7 이하가 2개이면, 4개 가져오라해도 2개만 줌
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # 0.7이상만 반환
    },
)

# 3. MultiQueryRetriever -> 질문을 llm을 통해 질문을 강화하고, 다시 넣음
"""
LLM이 질문을 3~5개로 변형:                                                                                          
원본: "LCEL이란?"
변환: "LangChain Expression Language란?",                                                                           
"LCEL 문법과 사용법은?",                                                                                      
"LCEL과 기존 Chain의 차이점은?"

각 변형 질문으로 검색 -> 결과 합집합(중복 제거) -> 재현율 크게 향상

단점 -> 검색할 때마다 llm을 한번 더 호출해야함 -> 속도 vs 정확도의 문제 
"""

from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,  # 질문 변환용 LLM
)
