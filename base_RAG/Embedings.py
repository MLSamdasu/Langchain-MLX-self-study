# 허깅페이스 로컬 임베딩 설정 
from langchain_community.embeddings import HuggingFaceEmbeddings

# 한국어 지원 임베딩 
embedding = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
    model_kwargs={"device":"mps"},
    encode_kwargs={"normalize_embeddings":True}
)

# 임베딩 테스트
text = "LangChain 은 LLM 애플리케이션 프레임워크 입니다. "
vector = embedding.embed_query(text)
print(f"벡터 차원: {len(vector)}")