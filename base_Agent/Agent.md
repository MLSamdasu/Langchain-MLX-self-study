### Agent
- Agent는 LLM이 스스로 판단하여 도구(Tool)를 선택하고, 실행하는 자율 시스템
- 체인 : 코드 작성 시점에 실행 순서가 확정됨 
- 에이전트 : 여러 도구가 주어지고, LLM이 입력을 보고 어떤 도루를 쓸지, 말지, 몇번 쓸지를 런타임에 직접 판단함 

### ReAct 루프 - Agent의 핵심 엔진 

> ReAct = Reasoning(추론) + Acting(행동)

- 각 단계
> Query -> (LLM Think -> Tool Select -> Execute -> Observe) -> Answer

Observe 단계에서 부족하다 생각되면 LLM Think로 다시 돌아감

- 각 단계별 설명
    - Query
        - 사용자 질문 입력
    - Think 
        - LLM이 질문 분석, 전략 수립
    - Tool Select
        - 도구 목록에서 적합한 것 선택
    - Execute
        - 선택한 도구 실행
    - Observe
        - 실행 결과 확인
    - Answer
        - 충분하면 답변 생성 

- 돌아갈 수 있다 -> 이게 핵심임(체인과 다른점)
    - 계속 돌아가면서 도구들을 쓰면서 정보를 모은 후 답변 

### tool 데코레이터 프로토타입 
- @tool 은 간편하지만, 단일 문자열 입력에 적합함. 
- 파라미터가 여러개면 아래 Pydantic 기반등 다른 방식을 주로 씀 

```python
from langchain_core.tools import tool
@tool
def 함수이름(파라미터: 타입) -> 반환 타입:
    """ 이 도구가 뭘 하는지 설명 """
    try:
        return 결과
    except Exception as e:
        return f"오류:{e}"
```


### pydantic 기반 structured tool
- 복잡한 입력을 받는 도구는 Pydantic 모델로 입력 스키마를 정의함 
    - 예시
    - 파일 읽기 -> 경로 + 인코딩 2개
    - DB 검색 -> 테이블명 + 조건 + 정렬 3개 
