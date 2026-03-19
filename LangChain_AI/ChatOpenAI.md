## 모델 호출 방식 2가지

### 1. 서버 경유
- ChatOpenAI_MLX.py 에서 사용한 방식
- Python → HTTP 요청 → mlx_lm.server → 모델 → HTTP 응답 → Python

### 2. 직접 호출
- 서버 없이 직접 호출함 
- MLXChatModel.py 에서 사용한 방식
- Python → mlx_lm.generate() → 모델 → 바로 결과
- mlx_lm.generate()는 일반함수 -> prompt | llm | parser 파이프라인에 못 끼워넣음 
    - LangChain의 BaseChatModel을 상속해서 커스텀 클래스를 만들어 씀 


#### ChatPromptTemplate에서 .from_template과 .from_messages의 차이
- from_template
    - 단일 문자열
    - 일반 LLM -> 하나의 입력에 대해 거의 일관된 하나의 출력만 나와야 하는 경우
    - 지시문 내에 텍스트로 역할 정의
    - 단순 번역, 키워드 추출처럼 직관적인 1:1 대응이 필요할때 주로 사용 
- from_messages
    - Chat Models(gpt, claude 등)에 주로 사용
    - 메시지 리스트(List of Tuples , Objects) -> []를 사용해야함 
    - 시스템, 유저, AI 역할로 구조적 분리
    - 대화의 맥락과 흐름을 파악하기 좋음 
    - 복잡한 페르소나가 필요한 경우 사용 
    - 최근에 표준에 가깝게 사용됨 