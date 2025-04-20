import os
import streamlit as st
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions
from dotenv import load_dotenv

# Load environment variables and set page config
load_dotenv()
st.set_page_config(
    page_title="YourBot: Article Research Tool",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main, .block-container {
        padding-top: 2rem;
        max-width: 95%;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        margin-top: 1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    h1 {
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    .source-box {
        padding: 0.5rem;
        border-left: 3px solid #FF4B4B;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# Main layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🤖 YourBot: Article Research Tool")

# Sidebar styling and layout
with st.sidebar:
    st.markdown("### 📰 News Article URLs")
    urls = []
    for i in range(3):
        url = st.text_input(
            f"URL {i + 1}",
            placeholder=f"Enter article URL {i + 1}",
            key=f"url_{i}"
        )
        urls.append(url)

    process_url_clicked = st.button("🚀 Process URLs")

# Initialize variables
index_dir = "./faiss_index"
os.makedirs(index_dir, exist_ok=True)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=500)

# URL Processing
if process_url_clicked:
    valid_urls = [url for url in urls if url.strip()]
    if not valid_urls:
        st.error("⚠️ Please provide at least one valid URL!")
    else:
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        try:
            # Load data
            status_placeholder.markdown('<div class="status-box">📥 Loading data from URLs...</div>',
                                        unsafe_allow_html=True)
            loader = UnstructuredURLLoader(urls=valid_urls)
            data = loader.load()
            progress_bar.progress(0.33)

            # Split data
            status_placeholder.markdown('<div class="status-box">✂️ Splitting text into chunks...</div>',
                                        unsafe_allow_html=True)
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(data)
            progress_bar.progress(0.66)

            # Create embeddings
            status_placeholder.markdown('<div class="status-box">🔄 Building embedding vectors...</div>',
                                        unsafe_allow_html=True)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore_gemini = FAISS.from_documents(docs, embeddings)
            FAISS.save_local(vectorstore_gemini, index_dir)
            progress_bar.progress(1.0)

            status_placeholder.success("✅ Processing complete! Ready for questions.")
            time.sleep(2)
            progress_bar.empty()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# Query handling with retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted)
)
def get_result(chain, query):
    return chain({"question": query}, return_only_outputs=True)


# Query interface
query = st.text_input("🔍 Ask your question:", placeholder="Enter your question about the articles...")

if query:
    if os.path.exists(index_dir):
        with st.spinner("🤔 Thinking..."):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = get_result(chain, query)

                st.markdown("### 💡 Answer")
                st.write(result["answer"])

                sources = result.get("sources", "")
                if sources:
                    st.markdown("### 📚 Sources")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        if source.strip():
                            st.markdown(f'<div class="source-box">{source}</div>', unsafe_allow_html=True)

            except google.api_core.exceptions.ResourceExhausted as e:
                st.error("⚠️ Resource limit reached. Please wait a moment and try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please process some URLs first!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Powered by Google LLM Flash & LangChain</p>",
    unsafe_allow_html=True
)