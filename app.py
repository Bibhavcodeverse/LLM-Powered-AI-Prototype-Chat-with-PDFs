import streamlit as st
import os
import shutil
from rag_pipeline import RAGSystem

# Page Config
st.set_page_config(page_title="Chat with PDFs", layout="wide")

st.title("🤖 Chat with PDFs Prototype")

# Initialize Session State
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# Sidebar - File Upload
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Save uploaded files to temp directory
                temp_dir = "temp_pdfs"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                os.makedirs(temp_dir)
                
                saved_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(file_path)
                
                # Ingest
                success = st.session_state.rag_system.ingest(saved_paths)
                
                if success:
                    # Initialize Chain
                    st.session_state.rag_system.get_chain()
                    st.session_state.processing_complete = True
                    st.success("✅ Documents processed successfully!")
                else:
                    st.error("❌ Failed to process documents.")
        else:
            st.warning("Please upload at least one PDF.")

# Chat Interface
if st.session_state.processing_complete:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Source Documents"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}**: {source.metadata.get('source', 'Unknown')} (Page {source.metadata.get('page', 0)})")
                        st.text(source.page_content[:200] + "...")

    # User Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chain = st.session_state.rag_system.chain
                    if chain:
                        response = chain.invoke({"question": prompt, "chat_history": []}) 
                        # Note: manual handling of history might be needed depending on chain config, 
                        # but ConversationBufferMemory handles it internally if configured correctly.
                        # However, passing empty list here to avoid duplication if memory is attached.
                        
                        answer = response.get("answer", "I don't know.")
                        sources = response.get("source_documents", [])
                        
                        st.markdown(answer)
                        
                        # Display sources for verification (Hallucination Check)
                        if sources:
                            with st.expander("📚 Source Documents"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1}**: {source.metadata.get('source', 'Unknown')} (Page {source.metadata.get('page', 0)})")
                                    st.text(source.page_content[:200] + "...")
                        else:
                             st.caption("No sourced documents met the similarity threshold.")

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                except Exception as e:
                    import traceback
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                    print(traceback.format_exc())
else:
    st.info("👈 Please upload and process documents to start chatting.")
