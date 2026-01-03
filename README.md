# askpdf

A full-stack PDF reading assistant that combines **Text-to-Speech (TTS)**, **RAG (Retrieval Augmented Generation)**, and **AI chat** capabilities. Upload a PDF, have it read aloud with synchronized text highlighting, and chat with your document using AI.

## 🌟 Features

### 📄 Reading & TTS
- **Unified Experience**: Seamlessly switch between reading the PDF and listening to chat responses
- **Intelligent Text Processing**: Robust sentence segmentation with support for Markdown and non-punctuated text
- **High-Quality TTS**: Local speech synthesis using [Kokoro-82M](https://github.com/hexgrad/kokoro)
- **Visual Tracking**: Synchronized sentence highlighting in PDF and message highlighting in Chat
- **Interactive Navigation**: Double-click any sentence in the PDF or any message in the Chat to start playback
- **Centralized Controls**: Unified player in the footer manages all audio sources (Speed 0.5x - 2.0x)

### 💬 RAG-Powered Chat
- **Semantic Search**: Ask questions about your PDF content
- **Vector Storage**: Document chunks indexed in Qdrant for fast retrieval
- **Conversational AI**: Chat with context from your document using local LLMs
- **Chat History**: Maintains conversation context for follow-up questions

### 🎨 Modern UI
- **Unified Navigation**: Double-click sentences or chat bubbles to start reading immediately
- **Dynamic Visual Feedback**: PDF sentence highlighting and Chat bubble illumination during playback
- **Resizable Chat Panel**: Drag to adjust the chat interface width (300-800px)
- **Auto-Scroll**: Both PDF and Chat automatically keep the active being-read content in view
- **Model Selection**: Centralized embedding model selection and dynamic LLM discovery

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Docker Compose                                 │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│    Frontend     │    Backend      │   RAG Service   │       Qdrant          │
│   (Next.js)     │    (FastAPI)    │    (FastAPI)    │   (Vector DB)         │
│   Port: 3000    │   Port: 8000    │   Port: 8001    │   Port: 6333          │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘
                                          │
                                          ▼
                            ┌─────────────────────────┐
                            │   DMR / Ollama / LLM    │
                            │   (OpenAI-compatible)   │
                            │      Port: 12434        │
                            └─────────────────────────┘
```

### Services Overview

| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | 3000 | Next.js React app with PDF viewer and chat UI |
| **Backend** | 8000 | FastAPI server for PDF processing and TTS |
| **RAG Service** | 8001 | FastAPI server for document indexing and AI chat |
| **Qdrant** | 6333 | Vector database for semantic search |
| **DMR/Ollama** | 12434 | Local LLM server (external, user-provided) |

## 📋 Prerequisites

- **Docker** and **Docker Compose**
- **Local LLM Server**: The app is configured to use **Docker Model Runner (DMR)** by default on port \`12434\`.
  - **Option A: DMR (Default)** - Built into Docker Desktop.
  - **Option B: Ollama** - Requires running on port `12434` or updating configuration.

### Required Models (on your LLM server)
- **LLM Model**: e.g., `ai/qwen3:latest` (DMR) or `llama3` (Ollama)
- **Embedding Model**: e.g., `ai/nomic-embed-text-v1.5:latest` (DMR) or `nomic-embed-text` (Ollama)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/raghu13590/askpdf.git
cd askpdf
```

### 2. Start Your Local LLM Server

The application expects an OpenAI-compatible API at \`http://localhost:12434\`.


**Option A: Using Docker Model Runner (DMR) - Recommended Default**
DMR is built into Docker Desktop and runs on port `12434` by default.

```bash
# Ensure Docker Desktop is running and DMR is enabled
```

**Installing Models in DMR**

DMR requires you to download and import models manually. Follow these steps to install the required models:

1. **Download the model files** from Hugging Face or your preferred source:
  - LLM Model (e.g., `ai/qwen3:latest`): [Qwen3 on Hugging Face](https://huggingface.co/Qwen/Qwen1.5-7B-Chat)
  - Embedding Model (e.g., `ai/nomic-embed-text-v1.5:latest`): [nomic-embed-text-v1.5 on Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

2. **Import models into DMR** using the DMR UI or CLI:
  - Open Docker Desktop, go to the DMR extension, and use the "Import Model" button to add the downloaded models.
  - Alternatively, use the DMR CLI (if available) to import models:
    ```bash
    dmr import <path-to-model-directory>
    ```

3. **Verify models are available** in the DMR UI under the "Models" tab.

> **Note:** Model names in the app (e.g., `ai/qwen3:latest`, `ai/nomic-embed-text-v1.5:latest`) must match the names you assign in DMR.

**Option B: Using Ollama**
Ollama runs on port `11434` by default. To use it with this app without changing code, start it on port `12434`:
```bash
# Start Ollama on the expected port
OLLAMA_HOST=0.0.0.0:12434 ollama serve

# In a new terminal, pull the models
ollama pull llama3
ollama pull nomic-embed-text
```

### 3. Start the Application

```bash
docker-compose up --build
```

### 4. Access the Application

- **Main App**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **RAG API**: http://localhost:8001
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📖 Usage

### Reading a PDF

1. **Select Embedding Model**: Choose an embedding model from the dropdown
2. **Upload PDF**: Click "Upload PDF" and select your file
3. **Wait for Processing**: The PDF is parsed, sentences extracted, and indexed for RAG
4. **Play Audio**: Click "Play" to start text-to-speech from the beginning
5. **Navigate**: Use playback controls or double-click any sentence in the PDF or any chat bubble to jump to it
6. **Adjust Voice**: Select different voice styles and adjust playback speed

### Chatting with Your PDF

1. **Select LLM Model**: Choose an LLM from the chat panel dropdown
2. **Ask Questions**: Type your question about the PDF content
3. **Get AI Answers**: The system retrieves relevant chunks and generates answers
4. **Continue Conversation**: Follow-up questions maintain context
5. **Read Out Loud**: Double-click any chat bubble to have the assistant's response (or your own question) read aloud

## 🛠️ Technology Stack

### Backend Service
| Technology | Purpose |
|------------|---------|
| **FastAPI** | Web framework for REST APIs |
| **PyMuPDF (fitz)** | PDF parsing with character-level coordinates |
| **spaCy** | NLP for sentence segmentation |
| **Kokoro** | Neural TTS with 82M parameters |

### RAG Service
| Technology | Purpose |
|------------|---------|
| **FastAPI** | Web framework |
| **LangChain** | LLM/Embedding integration |
| **LangGraph** | Stateful RAG workflow |
| **Qdrant Client** | Vector database operations |

### Frontend
| Technology | Purpose |
|------------|---------|
| **Next.js** | React framework |
| **Material-UI (MUI)** | UI components |
| **react-pdf** | PDF rendering |
| **react-markdown** | Chat message rendering |

## 📁 Project Structure

```
askpdf/
├── docker-compose.yml          # Multi-service orchestration
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py             # FastAPI app, upload & TTS endpoints
│       ├── pdf_parser.py       # PyMuPDF text extraction with coordinates
│       ├── nlp.py              # spaCy sentence segmentation
│       └── tts.py              # Kokoro TTS synthesis
├── rag_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # FastAPI app, index & chat endpoints
│   ├── rag.py                  # Document chunking & indexing
│   ├── agent.py                # LangGraph RAG workflow
│   ├── models.py               # LLM/Embedding model clients
│   └── vectordb/
│       ├── base.py             # Abstract vector DB interface
│       └── qdrant.py           # Qdrant adapter implementation
└── frontend/
    ├── Dockerfile
    ├── package.json
    └── src/
        ├── pages/
        │   └── index.tsx       # Main application page
        ├── components/
        │   ├── PdfUploader.tsx     # File upload with model selection
        │   ├── PdfViewer.tsx       # PDF rendering with overlays
        │   ├── PlayerControls.tsx  # Audio playback controls
        │   ├── ChatInterface.tsx   # RAG chat UI
        │   └── TextViewer.tsx      # Alternative text display
        └── lib/
            ├── api.ts          # Backend API client
            └── tts-api.ts      # TTS API client
```

## 📝 API Reference

### Backend Service (Port 8000)

#### `POST /api/upload`
Upload a PDF and extract sentences with bounding boxes.

**Request:** `multipart/form-data`
- `file`: PDF file
- `embedding_model`: Model name for RAG indexing

**Response:**
```json
{
  "sentences": [
    {
      "id": 0,
      "text": "First sentence.",
      "bboxes": [
        {"page": 1, "x": 72, "y": 700, "width": 50, "height": 12, "page_height": 792, "page_width": 612}
      ]
    }
  ],
  "pdfUrl": "/abc123.pdf"
}
```

#### `GET /api/voices`
List available TTS voice styles.

**Response:**
```json
{
  "voices": ["M1.json", "F1.json", "M2.json"]
}
```

#### `POST /api/tts`
Synthesize speech for text.

**Request:**
```json
{
  "text": "Text to synthesize",
  "voice": "M1.json",
  "speed": 1.0
}
```

**Response:**
```json
{
  "audioUrl": "/data/audio/tmp_xyz.wav"
}
```

### RAG Service (Port 8001)

#### `POST /index`
Index document text into vector database.

**Request:**
```json
{
  "text": "Full document text...",
  "embedding_model": "ai/nomic-embed-text-v1.5:latest",
  "metadata": {"filename": "document.pdf", "upload_id": "uuid"}
}
```

#### `POST /chat`
Chat with indexed documents.

**Request:**
```json
{
  "question": "What is this document about?",
  "llm_model": "ai/qwen3:latest",
  "embedding_model": "ai/nomic-embed-text-v1.5:latest",
  "history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ]
}
```

**Response:**
```json
{
  "answer": "This document discusses...",
  "context": "Retrieved chunks used for the answer..."
}
```

#### `GET /models`
Fetch available models from LLM server.

#### `GET /health`
Health check endpoint.

## 🔧 Configuration

### Environment Variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | Frontend | `http://localhost:8000` | Backend API URL |
| `NEXT_PUBLIC_RAG_API_URL` | Frontend | `http://localhost:8001` | RAG API URL |
| `RAG_SERVICE_URL` | Backend | `http://rag-service:8000` | Internal RAG service URL |
| `QDRANT_HOST` | RAG Service | `qdrant` | Qdrant hostname |
| `QDRANT_PORT` | RAG Service | `6333` | Qdrant port |
| `DMR_BASE_URL` | RAG Service | `http://host.docker.internal:12434` | LLM server URL (Change to `...:11434` for default Ollama) |

### Voice Styles

Voice styles (voices) are handled by the Kokoro engine. Available options are discovered dynamically from the system and populated in the UI dropdown.

### TTS Parameters

In `backend/app/tts.py`:
- `total_step`: Diffusion steps (default: 5) - higher = better quality, slower
- `speed`: Playback speed (0.5 - 2.0)

## 🔄 Data Flow

### PDF Upload Flow
```
User uploads PDF
  ↓
Backend: Save PDF → Extract text + coordinates (PyMuPDF)
  ↓
Backend: Split into sentences (spaCy)
  ↓
Backend: Map sentences to bounding boxes
  ↓
Backend: Trigger async RAG indexing
  ↓
RAG Service: Chunk text → Generate embeddings → Store in Qdrant
  ↓
Frontend: Display PDF with clickable sentence overlays
```

### Chat Flow
```
User asks question
  ↓
RAG Service: Embed question
  ↓
RAG Service: Search Qdrant for top-5 relevant chunks
  ↓
RAG Service: Build prompt (system + context + history + question)
  ↓
RAG Service: Call LLM via OpenAI-compatible API
  ↓
Frontend: Display markdown-rendered answer
```

### TTS Playback Flow
```
User clicks Play or double-clicks sentence
  ↓
Frontend: Request /api/tts with sentence text
  ↓
Backend: Kokoro synthesizes audio → WAV file
  ↓
Frontend: Play audio, highlight current sentence
  ↓
On audio end: Auto-advance to next sentence
```

## 🐳 Docker Details

The application uses Docker Compose with four services:

1. **frontend**: Next.js dev server with hot reload
2. **backend**: FastAPI with TTS models mounted (Supertonic cloned from HuggingFace at build)
3. **rag-service**: FastAPI with LangChain/LangGraph
4. **qdrant**: Official Qdrant image with persistent storage

### Volumes
- \`qdrant_data\`: Persistent vector storage
- Source directories mounted for development hot-reload

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project uses the following third-party technologies:
- [Kokoro](https://github.com/hexgrad/kokoro) - Text-to-speech model
- [spaCy](https://spacy.io/) - Natural language processing
- [LangChain](https://langchain.com/) - LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Stateful AI workflows
- [Qdrant](https://qdrant.tech/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Next.js](https://nextjs.org/) - React framework

## 🙏 Acknowledgments

- **hexgrad** for the amazing Kokoro-82M model
- **spaCy** for robust NLP capabilities
- **LangChain** team for the excellent LLM framework
- **Qdrant** for the powerful vector database
- The open-source community for all the amazing tools

## 📧 Contact

For questions, issues, or suggestions, please open an issue on the [GitHub repository](https://github.com/raghu13590/askpdf).
