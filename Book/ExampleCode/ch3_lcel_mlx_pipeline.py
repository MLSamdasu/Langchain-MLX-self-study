"""
Ch3.5 LCEL 체인 + MLX 실전 파이프라인

이 파일의 역할:
    - Ch2에서 배운 LCEL 파이프 연산자(|)를 실제 MLX 로컬 모델과 결합한다
    - 프롬프트 템플릿 → MLX 모델 → 출력 파서의 3단계 파이프라인을 구성한다
    - Ch2의 mock 함수 대신 진짜 MLX 모델이 들어가는 실전 예제

비유:
    Ch2에서는 종이 비행기(mock)로 날리는 연습을 했다면,
    이제는 진짜 비행기(MLX 모델)에 연료를 넣고 이륙시키는 단계!

    [프롬프트 템플릿]  →  [MLX 로컬 모델]  →  [출력 파서]
       (주문서)           (요리사)            (포장 해체)

사전 준비:
    1) MLX 서버 실행:
       mlx_lm.server --model mlx-community/Qwen3.5-9B-4bit --port 8080
    2) 필요 패키지:
       pip install langchain-openai langchain-core

실행 방법:
    cd Book/ExampleCode
    python ch3_lcel_mlx_pipeline.py
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1) MLX 로컬 모델 연결
#    - 3.3에서 배운 ChatOpenAI + base_url 방식
#    - 이것이 파이프라인의 "두뇌" 역할
# ============================================================

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",  # MLX 로컬 서버 주소
    api_key="not-needed",                 # 로컬이라 키 불필요
    model="mlx-community/Qwen3.5-9B-4bit",
    temperature=0.7,
)


# ============================================================
# 2) 프롬프트 템플릿 정의
#    - ChatPromptTemplate으로 주문서 양식을 만든다
#    - {변수} 자리에 나중에 실제 값이 들어간다
# ============================================================

# 예제 1: 코드 리뷰 프롬프트 (교재 예제)
review_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 시니어 Python 개발자입니다."
               "코드를 리뷰하고 개선점을 제안하세요."),
    ("human", "다음 코드를 리뷰해주세요:\n```\n{code}\n```"),
])

# 예제 2: 한국어 요약 프롬프트 (추가 연습용)
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 텍스트 요약 전문가입니다. 핵심만 3줄로 요약하세요."),
    ("human", "다음 내용을 요약해주세요:\n{text}"),
])


# ============================================================
# 3) 출력 파서 준비
#    - StrOutputParser: AIMessage → 순수 문자열 변환
#    - 택배 상자(AIMessage)에서 내용물(문자열)만 꺼내는 역할
# ============================================================

parser = StrOutputParser()


# ============================================================
# 4) LCEL 파이프 체인 조립!
#    - Ch2에서 배운 | 연산자로 3개를 연결한다
#    - Ch2와 달리 prompt_to_str 변환이 필요 없다!
#      (ChatOpenAI는 ChatPromptValue를 직접 받을 수 있으므로)
# ============================================================

# 코드 리뷰 체인: 프롬프트 → MLX 모델 → 문자열 파서
review_chain = review_prompt | llm | parser

# 요약 체인: 프롬프트 → MLX 모델 → 문자열 파서
summary_chain = summary_prompt | llm | parser


# ============================================================
# 5) 헬퍼 함수
# ============================================================

def print_separator(title: str) -> None:
    """구분선과 제목을 출력하는 헬퍼 함수"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ============================================================
# 6) 실행
# ============================================================

def main() -> None:
    """LCEL + MLX 실전 파이프라인 실행"""

    # ----------------------------------------------------------
    # 예제 1: 코드 리뷰 파이프라인 (교재 3.5 핵심 예제)
    # ----------------------------------------------------------
    print_separator("예제 1: 코드 리뷰 파이프라인")

    # invoke()에 딕셔너리를 넘긴다 → {code} 자리에 값이 들어감
    result = review_chain.invoke({
        "code": "def add(a,b): return a+b"
    })
    print(f"\n리뷰 결과:\n{result}")

    # ----------------------------------------------------------
    # 예제 2: 요약 파이프라인
    # ----------------------------------------------------------
    print_separator("예제 2: 요약 파이프라인")

    result = summary_chain.invoke({
        "text": (
            "LangChain은 대규모 언어 모델을 활용한 애플리케이션 개발 프레임워크입니다. "
            "LCEL이라는 표현 언어를 통해 프롬프트, 모델, 파서를 파이프 연산자로 연결하여 "
            "복잡한 AI 파이프라인을 간결하게 구성할 수 있습니다. "
            "Apple Silicon의 MLX 프레임워크와 결합하면 로컬에서도 강력한 AI 시스템을 "
            "구축할 수 있다는 것이 이 책의 핵심 메시지입니다."
        )
    })
    print(f"\n요약 결과:\n{result}")

    # ----------------------------------------------------------
    # 데이터 흐름 확인: 각 단계별 출력 타입 살펴보기
    # ----------------------------------------------------------
    print_separator("보너스: 각 단계별 데이터 타입 확인")

    # 1단계: 프롬프트만 실행 → ChatPromptValue 반환
    prompt_result = review_prompt.invoke({"code": "print('hello')"})
    print(f"\n1단계 (프롬프트 출력 타입): {type(prompt_result).__name__}")
    print(f"  내용: {prompt_result.to_string()[:100]}...")

    # 2단계: 프롬프트 + LLM → AIMessage 반환
    llm_result = (review_prompt | llm).invoke({"code": "print('hello')"})
    print(f"\n2단계 (LLM 출력 타입): {type(llm_result).__name__}")
    print(f"  content: {llm_result.content[:100]}...")

    # 3단계: 프롬프트 + LLM + 파서 → str 반환
    final_result = review_chain.invoke({"code": "print('hello')"})
    print(f"\n3단계 (최종 출력 타입): {type(final_result).__name__}")
    print(f"  값: {final_result[:100]}...")

    # ----------------------------------------------------------
    # 정리
    # ----------------------------------------------------------
    print_separator("핵심 정리")
    print("""
    [LCEL + MLX 실전 파이프라인 핵심]

    1. 체인 구성은 딱 한 줄:
       chain = prompt | llm | StrOutputParser()

    2. Ch2의 mock과 달라진 점:
       - prompt_to_str 변환이 필요 없음 (ChatOpenAI가 직접 처리)
       - 실제 MLX 모델이 응답을 생성함

    3. 데이터 흐름:
       dict → ChatPromptValue → AIMessage → str
       (입력)    (프롬프트)        (LLM응답)   (결과)

    4. 다음 단계 (3.6):
       이 체인을 여러 개 만들어서 RunnableParallel로
       병렬 실행 → 멀티 모델 라우팅 파이프라인!
    """)


if __name__ == "__main__":
    main()
