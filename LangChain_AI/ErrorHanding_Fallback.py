# 로컬 모델은 서버 다운, OOM(메모리 부족) 등의 문제가 발생 할 수 있음. 
# LangChain의 .with_fallbacks()를 활용하여 자동 폴백 체인을 구성함 

# 폴백 체인 : 9B 실패시 3B로 자동 전환 
from LangChain_AI.LCEL_MLX_review import review_prompt
robust_llm = llm_heavy.with_fallbacks([llm_light])

# 재시도 + 폴백 조합 
from langchain_core.runnables import RunnableConfig

"""
with_retry()는 일시적 네트워크 오류에 유효하고,
with_fallbacks()는 모델 자체의 장애에 대응함
-> 따라서 둘을 같이 조합해서 안정성을 확보해야함. 
"""
resilient_chain = (
    review_prompt
    | robust_llm.with_retry(
        stop_after_attempt = 3,
        wait_exponential_multiplier=1,
    )
    | StrOutputParser() 
)