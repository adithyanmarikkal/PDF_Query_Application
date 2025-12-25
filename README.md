# PDF Query Application 📚

A powerful RAG (Retrieval-Augmented Generation) application that allows you to upload PDF documents and ask questions about their content using AI. Built with Streamlit, LangChain, and Google Gemini.

## Features ✨

- **PDF Upload**: Upload single or multiple PDF files
- **Intelligent Question Answering**: Ask questions about your PDF content and get accurate AI-generated answers
- **Vector Search**: Uses FAISS for efficient similarity search
- **Local Embeddings**: HuggingFace embeddings run locally (no API costs for embeddings)
- **Google Gemini Integration**: Powered by Google's Gemini 2.5 Flash model for answer generation
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface

## Technology Stack 🛠️

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.5 Flash (via LangChain)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS
- **PDF Processing**: PyPDF
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter

## Prerequisites 📋

- Python 3.8 or higher
- Google API Key (for Gemini)

## Installation 🚀

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pdfQuery
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   
   To get a Google API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy and paste it into your `.env` file

## Usage 💡

1. **Start the application**
   ```bash
   streamlit run main.py
   ```

2. **Upload PDFs**
   - Click on the sidebar menu
   - Upload one or more PDF files
   - Click "Submit & Process" to process the documents

3. **Ask Questions**
   - Type your question in the text input field
   - The AI will search through your PDFs and provide relevant answers

## How It Works 🔍

1. **PDF Processing**: Extracts text from uploaded PDF files
2. **Text Chunking**: Splits text into manageable chunks (1000 characters with 200 character overlap)
3. **Embedding Generation**: Creates vector embeddings using HuggingFace models (runs locally)
4. **Vector Storage**: Stores embeddings in FAISS index for fast similarity search
5. **Question Answering**: 
   - Converts user question to embeddings
   - Searches for similar chunks in FAISS index
   - Sends relevant chunks to Google Gemini for answer generation
   - Returns AI-generated answer based on document context

## Project Structure 📁

```
pdfQuery/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not in git)
├── .gitignore          # Git ignore file
├── README.md           # This file
├── faiss_index/        # FAISS vector database (generated)
└── venv/               # Virtual environment (not in git)
```

## Configuration ⚙️

You can modify the following parameters in `main.py`:

- **Chunk Size**: Default is 1000 characters (line 31)
- **Chunk Overlap**: Default is 200 characters (line 32)
- **Embedding Model**: Default is `sentence-transformers/all-MiniLM-L6-v2` (line 42)
- **LLM Model**: Default is `gemini-2.5-flash` (line 56)
- **Temperature**: Default is 0.3 for more focused answers (line 57)

## Troubleshooting 🔧

### Common Issues

1. **"GOOGLE_API_KEY not found" error**
   - Ensure your `.env` file exists and contains the API key
   - Verify the key is valid

2. **"Please process the PDF first" error**
   - Upload a PDF file first
   - Click "Submit & Process" before asking questions

3. **Slow processing**
   - First run downloads the HuggingFace model (~80MB)
   - Large PDFs take longer to process
   - Consider reducing chunk size for faster processing

4. **Memory issues**
   - For very large PDFs, consider processing them in batches
   - Reduce chunk size or increase chunk overlap

## Dependencies 📦

Key dependencies include:
- `streamlit` - Web interface
- `langchain` - LLM framework
- `langchain-google-genai` - Google Gemini integration
- `langchain-huggingface` - HuggingFace embeddings
- `faiss-cpu` - Vector similarity search
- `pypdf` - PDF text extraction
- `python-dotenv` - Environment variable management

See `requirements.txt` for complete list.

## Security 🔒

- Never commit your `.env` file or API keys to version control
- The `.gitignore` file is configured to exclude sensitive files
- Keep your Google API key secure and don't share it

## Performance Tips 🚀

- **Embeddings**: Run locally on CPU, no API costs
- **LLM Calls**: Only made when asking questions (uses Google API)
- **FAISS Index**: Saved locally for faster subsequent queries
- **Model Caching**: HuggingFace models are cached after first download

## Future Enhancements 🌟

Potential improvements:
- Support for more document formats (DOCX, TXT, etc.)
- Conversation history and context
- Multiple language support
- Custom embedding models
- Export answers to file
- Advanced search filters



---

**Built with ❤️ using Streamlit, LangChain, and Google Gemini**
