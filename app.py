import os
import tempfile
import streamlit as st
import chromadb
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# 1. Page Configuration & Environment Setup
st.set_page_config(page_title="AI Context Engine", page_icon="⚡", layout="wide")
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found. Please check your .env file.")
    st.stop()

# 2. Initialize Clients
client = genai.Client(api_key=api_key)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="context_notes")

# 3. Helper Functions for Processing Files
def extract_text_from_file(uploaded_file) -> str:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    if ext in [".txt", ".md"]:
        return uploaded_file.read().decode("utf-8")
        
    elif ext == ".pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    elif ext in [".mp3", ".wav", ".m4a"]:
        # Save temp file for Gemini API upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        st.toast(f"🎙️ Transcribing audio: {uploaded_file.name}...")
        uploaded_gemini_file = client.files.upload(file=tmp_path)
        
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=["Provide a complete, verbatim transcript of this audio.", uploaded_gemini_file]
        )
        os.remove(tmp_path)
        return response.text

    return ""

def process_and_index_file(uploaded_file):
    raw_text = extract_text_from_file(uploaded_file)
    if not raw_text.strip():
        st.warning(f"Could not extract text from {uploaded_file.name}")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)

    embeddings = []
    ids = []

    for idx, chunk in enumerate(chunks):
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk,
        )
        embeddings.append(response.embeddings[0].values)
        ids.append(f"{uploaded_file.name}_chunk_{idx}")

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    st.success(f"✅ Successfully indexed `{uploaded_file.name}` ({len(chunks)} chunks)")

# 4. Streamlit UI Layout
st.title("⚡ AI Context Engine")
st.caption("Upload documents or meeting audio to query your private knowledge base.")

# Sidebar for File Uploads
with st.sidebar:
    st.header("📄 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload files (.txt, .pdf, .mp3, .wav)", 
        type=["txt", "pdf", "mp3", "wav", "m4a"],
        accept_multiple_files=True
    )
    
    if st.button("Process Files", type="primary"):
        if uploaded_files:
            with st.spinner("Indexing documents..."):
                for file in uploaded_files:
                    process_and_index_file(file)
        else:
            st.warning("Please select at least one file to upload.")
            
    st.divider()
    if st.button("🗑️ Clear Vector Database"):
        chroma_client.delete_collection("context_notes")
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & Response Handling
if user_query := st.chat_input("Ask a question about your uploaded context..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Fast-path for simple greetings
    if user_query.strip().lower() in ["hi", "hello", "hey"]:
        answer = "Hello! Upload a file or ask me any question about your indexed context."
    else:
        # Retrieve vector context
        query_response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=user_query,
        )
        query_embedding = query_response.embeddings[0].values

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"]
        )

        retrieved_chunks = results["documents"][0] if results["documents"] else []
        context_str = "\n---\n".join(retrieved_chunks)

        prompt = f"""
        You are an AI assistant. Answer the question using ONLY the context provided below.
        If the answer is not contained in the context, respond with "I don't have that information in the notes."

        CONTEXT:
        {context_str}

        USER QUESTION:
        {user_query}
        """

        # Generate LLM response
        chat = client.chats.create(model="gemini-3.6-flash")
        response = chat.send_message(prompt)
        answer = response.text

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})