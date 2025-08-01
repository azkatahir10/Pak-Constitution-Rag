# 🇵🇰 Pakistan Constitution Q&A App

A simple RAG (Retrieval-Augmented Generation) based Question Answering application built using **LangChain**, **HuggingFace Transformers**, and **Streamlit**. This app allows users to ask questions about the Constitution of Pakistan and get accurate answers based on real constitutional text.

---

## 🧠 Features

- 📖 Loads and indexes the Constitution of Pakistan (in `.txt` format)
- 🔍 Semantic search using **FAISS**
- 🤖 Question-answering with HuggingFace LLM (e.g., `google/flan-t5-base`)
- 🧵 Built on LangChain for efficient text processing and chaining
- 💻 Simple UI built using Streamlit
- 🔐 API Key-based access to Hugging Face models

---

## 🗂 Project Structure

```

pak-constitution-rag/
│
├── constitution/
│   └── pak\_constitution.txt      # Raw text of the constitution
│
├── app.py                        # Streamlit frontend
├── ingest.py                     # Embedding and vector store creator
├── .env                          # HuggingFace API key
├── requirements.txt              # Project dependencies
└── README.md                     # You're here!

````

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pak-constitution-rag.git
cd pak-constitution-rag
````

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your HuggingFace token

* Go to [Hugging Face Tokens](https://huggingface.co/settings/tokens)
* Create a **read access token**
* Create a `.env` file and add:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## 🧪 Run the App

### Step 1: Ingest the Data

This step will create the FAISS vector database from the constitution text.

```bash
python ingest.py
```

### Step 2: Launch the Streamlit App

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to use the app.

---

## ❓ Sample Questions You Can Ask

* What are the fundamental rights in the Constitution?
* What is the role of Islam in Pakistan's Constitution?
* How is the National Assembly formed?
* What does the Constitution say about the rights of minorities?

---

## 🛠 Built With

* [LangChain](https://www.langchain.com/)
* [Hugging Face Transformers](https://huggingface.co/)
* [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
* [Streamlit](https://streamlit.io/)

---

## ✅ Future Improvements

* Add support for Urdu input
* Enable PDF or CSV uploads for new legal texts
* Improve LLM with fine-tuning or use OpenAI 


