from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    loaded_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300, separators=['\n\n','\n'," "])
    documents = text_splitter.split_documents(loaded_documents)
    return documents