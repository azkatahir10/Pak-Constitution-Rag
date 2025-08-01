import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_core.documents import Document

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Validate configuration
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("❌ Missing HuggingFace API token. Please check your .env file")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Pakistan Constitution Q&A", layout="wide")
st.title("🇵🇰 Pakistan Constitution Expert")
st.caption("Get accurate answers from the official constitutional text")

@st.cache_resource(show_spinner="Building knowledge base...")
def load_constitution_db():
    try:
        with open("constitution/pak_constitution.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        return FAISS.from_documents(docs, embeddings)
    
    except Exception as e:
        st.error(f"❌ Error loading constitution: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner="Initializing AI model...")
def load_llm():
    try:
        return HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            model_kwargs={
                "temperature": 0.1,
                "max_length": 512,
                "task": "text2text-generation"  # Explicitly set the task
            }
        )
    except Exception as e:
        st.error(f"❌ Failed to load AI model: {str(e)}")
        st.stop()

# Initialize components
vector_db = load_constitution_db()
llm = load_llm()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# User interface
question = st.text_input(
    "Enter your constitutional question:",
    placeholder="e.g. What is the official name of Pakistan?"
)

if question:
    with st.spinner("🔍 Analyzing constitutional provisions..."):
        try:
            result = qa_chain({"query": question})
            
            if result["result"]:
                st.subheader("Answer:")
                st.write(result["result"])
                
                with st.expander("📜 View Relevant Constitutional Text"):
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.markdown(f"**Excerpt {i}:**")
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        st.markdown("---")
            else:
                st.warning("No relevant answer found in the constitution.")
                
        except Exception as e:
            st.error(f"""
            ⚠️ Could not generate answer. Please try:
            1. A different question
            2. Checking your internet connection
            3. Trying again later
            
            Error: {str(e)}
            """)

# Sample questions
st.sidebar.markdown("### Try these questions:")
sample_qs = [
    "What is the official name of Pakistan?",
    "What are the fundamental rights?",
    "What does Article 25 say about equality?",
    "What constitutes high treason?"
]

for q in sample_qs:
    if st.sidebar.button(q):
        question = q