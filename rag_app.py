import os
import streamlit as st
from operator import itemgetter

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from langchain_classic.retrievers import (
    MultiQueryRetriever
)

# ------------------------
# Config
# ------------------------

load_dotenv()

st.set_page_config(
    page_title="Multi PDF RAG",
    page_icon="📚"
)

st.title("📚 Multi PDF RAG Chatbot")

DB_DIR = "./chroma_db"
UPLOAD_DIR = "./uploaded_pdfs"

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------
# Session State
# ------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "all_chats" not in st.session_state:
    st.session_state.all_chats = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# ------------------------
# Models
# ------------------------

def get_llm():
    """
    Pick the chat model from the LLM_PROVIDER env var.

    groq         -> free tier, runs Llama 3.3 70B (default)
    google       -> free tier, Gemini
    ollama       -> fully local, no key needed
    huggingface  -> needs paid HF Inference Providers credits
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model_name = os.getenv("LLM_MODEL")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name or "llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=512
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.0-flash",
            temperature=0.1,
            max_output_tokens=512
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name or "llama3.2",
            temperature=0.1
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        llm = HuggingFaceEndpoint(
            repo_id=model_name or "meta-llama/Llama-3.3-70B-Instruct",
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.05,
            max_new_tokens=512
        )
        return ChatHuggingFace(llm=llm)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


@st.cache_resource
def load_models():

    model = get_llm()

    # Embeddings run locally, so they are free
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return model, embeddings

model, embeddings = load_models()

# ------------------------
# Build RAG
# ------------------------

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_rag(pdf_paths):

    all_docs = []

    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    # Start fresh so old uploads don't pile up as duplicates
    try:
        Chroma(
            collection_name="multi_pdf_rag",
            embedding_function=embeddings,
            persist_directory=DB_DIR
        ).delete_collection()
    except Exception:
        pass

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="multi_pdf_rag",
        persist_directory=DB_DIR
    )

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
Generate three alternative versions
of the user's question to help retrieve
relevant document chunks.
Return one question per line, with no
numbering and no extra text.

Question:
{question}
"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        model,
        prompt=query_prompt
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful PDF assistant.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
    )

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history")
        }
        | prompt
        | model
        | StrOutputParser()
    )

    st.session_state.rag_chain = rag_chain

# ------------------------
# Sidebar
# ------------------------

with st.sidebar:

    if st.button("➕ New Chat"):
        if st.session_state.messages:
            st.session_state.all_chats.append(
                st.session_state.messages.copy()
            )
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.header("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        file_names = [f.name for f in uploaded_files]
        already_indexed = st.session_state.get("indexed_files")

        if file_names != already_indexed:

            with st.status("Processing PDFs...", expanded=True) as status:

                pdf_paths = []

                for file in uploaded_files:

                    dest = os.path.join(
                        UPLOAD_DIR,
                        file.name
                    )

                    with open(dest, "wb") as f:
                        f.write(file.getbuffer())

                    pdf_paths.append(dest)

                st.write("🔍 Indexing chunks...")

                build_rag(pdf_paths)

                st.write("✅ Ready!")

                status.update(
                    label="PDFs loaded successfully!",
                    state="complete"
                )

                st.session_state.indexed_files = file_names

        else:

            for f_name in file_names:
                st.write(f"✅ {f_name}")

            st.caption(
                "Already indexed — ask away!"
            )

    st.divider()

    if st.session_state.all_chats:

        st.subheader("🕘 Past Chats")

        for i, chat in enumerate(
            reversed(st.session_state.all_chats)
        ):

            first_msg = next(
                (
                    m["content"]
                    for m in chat
                    if m["role"] == "user"
                ),
                "Chat"
            )

            label = (
                first_msg[:40] + "..."
                if len(first_msg) > 40
                else first_msg
            )

            if st.button(
                f"💬 {label}",
                key=f"past_chat_{i}"
            ):

                if st.session_state.messages:
                    st.session_state.all_chats.append(
                        st.session_state.messages.copy()
                    )

                st.session_state.messages = chat
                st.rerun()

# ------------------------
# Main Chat
# ------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.messages:

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

question = st.chat_input(
    "Ask about your PDFs..."
)

if question:

    if st.session_state.rag_chain is None:

        st.error(
            "Please upload PDF documents first."
        )

        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    history = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages
    ])

    with st.spinner("Thinking..."):

        try:
            answer = st.session_state.rag_chain.invoke(
                {
                    "question": question,
                    "history": history
                }
            )
        except Exception as e:
            st.session_state.messages.pop()
            msg = str(e)
            if "402" in msg or "Payment Required" in msg:
                st.error(
                    "This provider is out of credits. Set "
                    "LLM_PROVIDER=groq (free) in your .env."
                )
            elif "401" in msg or "api_key" in msg.lower():
                st.error(
                    "Missing or invalid API key. Check your .env file."
                )
            else:
                st.error(f"LLM error: {msg}")
            st.stop()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()