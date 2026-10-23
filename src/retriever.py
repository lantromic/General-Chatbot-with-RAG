from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
import os

embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

def hybrid_retriever_reranked(chunks):
    if os.path.exists("chroma_db") and os.listdir("chroma_db"):
        vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.7, 0.3])

    compressor = FlashrankRerank(top_n=10)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

    return compression_retriever
