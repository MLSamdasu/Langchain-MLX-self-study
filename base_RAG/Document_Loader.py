from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export
from langchain_community.document_loaders import(
    PyPDFLoader, 
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)

# PDF 로딩 (페이지별 분리)
pdf_loader = PyPDFLoader("my_document.pdf")
pdf_docs = pdf_loader.load() # List[Document]

# 디렉토리 내 모든 .txt 로딩
dir_loader = DirectoryLoader(
    "./docs/",
    glob="**/*.txt",
    loader_cls=TextLoader, 
)
all_docs = dir_loader.load()

# Document 객체 구조
print(pdf_docs[0].page_content[:200]) # 텍스트
print(pdf_docs[0].metadata) # {source, page, ...}