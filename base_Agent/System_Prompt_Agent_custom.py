# 시스템 프롬프트 Agent 커스터 마이징
# 실전에서 Agent의 안정성의 80%를 결정함

"""
create_agent는 내부적으로 LangGraph 위에서 동작함,
시스템 프롬프트가 두가지 역할을 함
1. Tool 선택 가이드라인 -> 어떤 상황에서 어떤 Tool을 쓸지 명시적 규칙 제공
2. 출력 형식/언어 제약 -> 한국어로 답변하라고 할때, 없으면 영어로 답하는 경우가 빈번함

LangChain 1.0의 create_agent에서 prompt는 SystemMessage 또는 str을 받음
-> LangGraph prebuilt의 유사한 역할인데 LangGraph 방식이 deprecated되면서 통합된것임.

그래서
create_agent(model=llm, tools=tools, prompt=SystemMessage(...))
로 사용해야함
"""

from langchain_core.messages import SystemMessage

# 시스템 프롬프트로 Agent 역할 정의
agent = create_agent(
    model=llm,
    tools=tools,
    prompt=SystemMessage(
        content=(
            "당신은 Python 전문 개발 어시스턴트입니다.\n"
            "- 기술 문서 관련 질문은 search_docs를 사용\n"
            "- 계산이 필요하면 calculate를 사용\n"
            "한국어로 답변하세요."
        )
    ),
)
