# 허깅페이스 로컬 임베딩 설정 
from langchain_community.embeddings import HuggingFaceEmbeddings

# 한국어 지원 임베딩 
embedding = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
    model_kwargs={"device":"mps"},
    encode_kwargs={"normalize_embeddings":True}
)