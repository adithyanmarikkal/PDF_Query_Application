import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Google Gemini API
# Ensure GOOGLE_API_KEY is in your .env file
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")

def get_pdf_text(pdf_docs):
    """ Extract text from PDF """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """ Split text into chunks """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """ 
    Create embeddings using HuggingFace (runs locally, free).
    Model: all-MiniLM-L6-v2 is small and fast for CPUs.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Use Google Gemini for answer generation."""
    from dotenv import load_dotenv
    import os

    load_dotenv()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",          
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY"),  
    )

    # DEBUG: check what model is actually used
    print("DEBUG: Using Gemini model:", llm.model)

    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def user_input(user_question):
    """ Handle user question """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        st.write("Reply: ", response["output_text"])
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure you have processed the PDF first by clicking 'Submit & Process' on the sidebar.")

def main():
    st.set_page_config("PDF Query Application", page_icon=":books:", layout="wide")
    st.header("PDF Query Application")

    user_question = st.text_input("Ask a Question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.warning("Please upload a PDF first.")

if __name__ == "__main__":
    main()