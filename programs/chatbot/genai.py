#pip install streamlit pandas requests langchain langchain-community langchain-cohere cohere python-dotenv faiss-cpu  
#pip install openpyxl
# need to create new xlsx file with name a.xlsx

import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_cohere import CohereEmbeddings, ChatCohere

# Load environment variables from .env
load_dotenv()

def read_text_from_excel(excel_path):
    if not os.path.exists(excel_path):
        st.error("Excel file not found.")
        return None
    
    try:
        df = pd.read_excel(excel_path)
        text_data = "\n".join(df.astype(str).apply(lambda row: ' '.join(row), axis=1))
        return text_data
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

@st.cache_resource(show_spinner="Embedding Excel content...")
def create_retriever(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vector_store = FAISS.from_texts(texts, embeddings, normalize_L2=True)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def main():
    st.title("📘 NITTE University FEWD Result")
    st.write("Type ur USN.")

    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        st.error("Cohere API key not found. Please set it in your .env file.")
        return

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    if uploaded_file:
        with open("uploaded_data.xlsx", "wb") as f:
            f.write(uploaded_file.read())

        text = read_text_from_excel("a.xlsx")
        if not text:
            return

        retriever = create_retriever(text)
        st.session_state.retriever = retriever

        llm = ChatCohere(model="command-r-plus", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        user_input = st.text_input("Enter your query:")
        if user_input:
            with st.spinner("Searching..."):
                response = qa_chain.invoke(user_input)
                st.markdown("**Bot:** " + response["result"])
    else:
        st.info("Please upload an Excel file to begin.")

if __name__ == "__main__":
    main()
