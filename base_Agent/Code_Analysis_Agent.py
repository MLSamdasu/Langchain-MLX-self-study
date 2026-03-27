# 여기서 run_python, list_files, file_reader 3개의 tool을 조합하는 예시를 보여주는데,
"""
@tool
def run_python(code:str) -> str:
    ""Python 코드를 실행하고 결과를 반환합니다.""       <- 3따옴표로 해야함, 주석처리때문에 2개로 작성
    r = subprocess.run(["python3", "-c", code], ...)
"""
# 이건 학습/실험 환경에서만 써야됨 -> LLM이 생성한 코드를 subprocess로 실행하는건 임의코드 실행(RCE)취약점.
# 프로덕션에서는
# 1. Docker 샌드박스 안에서 실행
# 2. 화이트리스트 방식으로 허용가능한 모듈만 import 가능하게 제한
# 3. timeout 은 반드시 설정 (여기서는 30초로 하였음)


@tool
def run_python(code: str) -> str:
    """
    Python 코드를 실행하고 결과를 반환합니다.
    간단한 스크립트, 데이터 분석에 사용하시오.
    """
    import subprocess

    try:
        r = subprocess.run(
            ["python3", "-c", code], capture_output=True, text=True, timeout=30
        )
        output = r.stdout or r.stderr
        return output[:2000]
    except subprocess.TimeoutExpired:
        return "실행 시간 초과 (30초)"


@tool
def list_files(directory: str) -> str:
    """
    디렉토리 내 파일 목록을 반환합니다.
    """
    import os

    try:
        files = os.listdir(directory)
        return "\n".join(files[:50])
    except Exception as e:
        return f"오류: {e}"


# 코드 분석 Agent
code_agent = create_agent(
    model=llm,
    tools=[run_python, list_files, file_reader],
)
