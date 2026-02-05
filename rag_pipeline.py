import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer' # Key for the final answer
        )

    def ingest(self, pdf_file_paths):
        """
        Ingests a list of PDF file paths.
        """
        documents = []
        for file_path in pdf_file_paths:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Create VectorStore
        if chunks:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            return True
        return False

    def get_chain(self):
        if not self.vectorstore:
            return None
            
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        from langchain.prompts import PromptTemplate

        custom_template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end. 
If the answer is not in the context, say "I do not have enough information to answer this based on the provided documents." and do not try to make up an answer.

Context:
{context}

Question: {question}
Helpful Answer:"""
        
        prompt = PromptTemplate(template=custom_template, input_variables=["context", "question"])

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        return self.chain
