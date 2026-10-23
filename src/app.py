from dotenv import load_dotenv
import streamlit as st
import os
import json
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from ingestion import load_and_chunk
from retriever import hybrid_retriever_reranked

load_dotenv()
# Set API Key
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🧭 Navigation & Architecture")
app_mode = st.sidebar.radio(
    "Choose RAG Strategy:",
    ["Hybrid RAG (Vector + BM25 + FlashRank)", "Vectorless Tree RAG (PageIndex)", "No RAG Chatbot"]
)

#start---
if app_mode == "No RAG Chatbot":
    st.title("No RAG Chatbot")
    st.caption("A simple chatbot without any retrieval mechanism.")

    if "display_msgs_no_rag" not in st.session_state:
        st.session_state["display_msgs_no_rag"] = []

    @st.cache_resource
    def load_no_rag_pipeline():
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.1)
    
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a general AI assistant, try to answer the user's question as best as you can."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        chain = prompt | llm | StrOutputParser()
        return chain

    chain = load_no_rag_pipeline()

    for message in st.session_state["display_msgs_no_rag"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question...")
    if user_input:
        history = list(st.session_state["display_msgs_no_rag"]) 

        st.session_state["display_msgs_no_rag"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = chain.invoke({
                "chat_history": history,
                "question": user_input,
            })
            st.markdown(response)
            
        response = chain.invoke({
            "chat_history": history,
            "question": user_input,
        })

        st.session_state["display_msgs_no_rag"].append({"role": "assistant", "content": response})

#end---

# ==========================================
# SECTION 1: HYBRID RAG
# ==========================================
if app_mode == "Hybrid RAG (Vector + BM25 + FlashRank)":
    st.title("📄 Hybrid RAG Chatbot")
    st.caption("Traditional chunking with ChromaDB, BM25 keyword search, and FlashRank reranking.")

    if "display_msgs_hybrid" not in st.session_state:
        st.session_state["display_msgs_hybrid"] = []

    @st.cache_resource
    def load_hybrid_pipeline():
        chunks = load_and_chunk("data/stuff.pdf")
        retriever = hybrid_retriever_reranked(chunks)
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an enterprise AI assistant. Answer the user's question "
                       "accurately using ONLY the provided context.\n\nContext:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        chain = prompt | llm | StrOutputParser()
        return retriever, chain

    retriever, chain = load_hybrid_pipeline()

    for message in st.session_state["display_msgs_hybrid"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a hybrid RAG question about your document...")
    if user_input:
        history = list(st.session_state["display_msgs_hybrid"]) 

        st.session_state["display_msgs_hybrid"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            docs = retriever.invoke(user_input)
            with st.expander("🔍 View Retrieved Chunks (Debug)"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}** (Source: page {doc.metadata.get('page', 'unknown')})")
                    st.text(doc.page_content[:500] + "...")
            
            context = "\n\n".join(doc.page_content for doc in docs)
            
            response = chain.invoke({
                "context": context,
                "chat_history": history,
                "question": user_input,
            })

        st.session_state["display_msgs_hybrid"].append({"role": "assistant", "content": response})


# ==========================================
# SECTION 2: VECTORLESS TREE RAG (PAGEINDEX)
# ==========================================
elif app_mode == "Vectorless Tree RAG (PageIndex)":
    st.title("🌲 Vectorless Tree RAG (questions about document is preferred)")

    if "display_msgs_vectorless" not in st.session_state:
        st.session_state["display_msgs_vectorless"] = []

    @st.cache_resource
    def load_genai_client():
        return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    @st.cache_data
    def load_tree_data():
        path = "data/stuff_tree.json" if os.path.exists("data/stuff_tree.json") else "../data/stuff_tree.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    genai_client = load_genai_client()
    tree_data = load_tree_data()

    for message in st.session_state["display_msgs_vectorless"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a vectorless tree question about your document...")
    if user_input:
        history = list(st.session_state["display_msgs_vectorless"])

        st.session_state["display_msgs_vectorless"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Reasoning through hierarchical tree structure..."):
                prompt = f"""
You are an expert technical document analyst.
Below is the hierarchical tree structure and summaries of an ISO standard document:

{json.dumps(tree_data, indent=2)}

User Question: {user_input}

Instructions:
1. Locate the exact section, node, or sub-clause corresponding to the query.
2. Provide a clear, structured response addressing the user's question accurately based on the tree summaries.
"""
                response = genai_client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                )
                response_text = response.text

        st.session_state["display_msgs_vectorless"].append({"role": "assistant", "content": response_text})