# app.py
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

# -- Set your Gemini API key and model --
os.environ["GOOGLE_API_KEY"] = "AIzaSyAgEitj_uAy5UpNmxNl5RX8Sdh6BAQw0C4"

# -- Streamlit UI setup --
st.set_page_config(page_title="PDF QA App", layout="wide")
st.title("📄 Ask Questions from PDF using Gemini + RAG")

# -- Initialize chat history --
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -- File uploader and question input --
pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Ask a question from the PDF")

if pdf:
    # Step 1: Save uploaded PDF
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf.read())

    # Step 2: Load PDF as documents
    loader = PyMuPDFLoader("uploaded.pdf")
    documents = loader.load()

    # Step 3: Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(documents, embeddings)

    # Step 4: Set up retriever and Gemini LLM
    retriever = db.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Step 5: Run query and store chat
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain({"query": query})
            answer = result["result"]

            # Append to chat history
            st.session_state.chat_history.append((query, answer))

            # Display the latest answer
            st.markdown("### 🔍 Answer")
            st.write(answer)

            with st.expander("📚 Source Documents"):
                for doc in result["source_documents"]:
                    st.markdown(f"**Page Content:** {doc.page_content[:500]}...")

    # Step 6: Display chat history
    if st.session_state.chat_history:
        st.markdown("## 💬 Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
