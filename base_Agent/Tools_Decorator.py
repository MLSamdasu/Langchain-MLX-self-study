"""
Tool -> ReAct 루프에서 Tool Select -> Execute 단계에 들어가는 "도구"를 직접 만드는 방법
@tool 데코레이터로 감싸기만 하면됨
tool로 감싸면 3가지가 추가됨
1. name -> 도구이름 -> llm이 사용할 도구 이름 지칭
2. description -> 도구 설명 -> 이 도구가 뭘 하는지 판단
3. 입력 스키마 -> 입력 형식 -> 뭘 넣어서 사용하는지 이해


try-except 를 안쓰면 llm이 멈춰버림, except를 써야 오류로 알아듣고 다른 도구 사용함

"""

from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """
    수학 계산을 수행합니다.
    사칙연산, 거듭제곱 등 Python 수식을 입력하시오
    예 : '2 + 3 * 4', '2**10'
    """
    try:
        result = eval(expression)
        return f"결과: {result}"
    except Exception as e:
        return f"계산 오류 : {e}"


@tool
def search_web(query: str) -> str:
    """
    웹 검색을 수행하여 최신의 정보를 가져옴
    검색어를 입력하면 관련 정보를 반환함.
    """
    # 실제로는 Tavily, SerpAPI 등을 사용
    return f"'{query} 에 대한 검색 결과...'"


# Tool 메타정보 확인
if __name__ == "__main__":
    print(calculate.name)
    print(calculate.description)
