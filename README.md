# PDF to Speech Reader

A full-stack application that converts PDF documents into natural-sounding speech using AI-powered text-to-speech (TTS) technology. Upload a PDF, and listen to it read aloud with professional voice synthesis powered by Supertonic.

## 🌟 Features

- **PDF Upload & Parsing**: Extract text from any PDF document
- **Intelligent Text Processing**: Automatically split text into sentences using NLP
- **High-Quality TTS**: Generate natural-sounding speech using Supertonic's AI voice models
- **Interactive Player**: Play, pause, resume, and navigate through sentences
- **Visual Feedback**: Highlight the currently playing sentence with auto-scroll
- **Seamless Experience**: Modern, responsive UI built with Next.js and Material-UI

## 🏗️ Architecture

The application consists of two main components:

### Backend (Python/FastAPI)
- **PDF Parsing**: Uses `pdfminer.six` to extract text from PDF files
- **NLP Processing**: Leverages spaCy (`en_core_web_sm`) to split text into sentences
- **Text-to-Speech**: Integrates Supertonic's ONNX models for voice synthesis
- **API Endpoints**:
  - `POST /api/upload` - Upload PDF and extract sentences
  - `POST /api/tts` - Synthesize speech for a given sentence

### Frontend (Next.js/React)
- **Modern UI**: Built with Next.js 14, React 18, and Material-UI 6
- **Component-Based**: Modular architecture with reusable components
- **Interactive Controls**: Player controls for playback management
- **Static Export**: Optimized for deployment as static files

## 📋 Prerequisites

- Docker and Docker Compose (recommended)
- OR:
  - Python 3.11+
  - Node.js 20+
  - Git LFS (for downloading TTS models)

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd document-reader-supertonic
   ```

2. **Build and run**:
   ```bash
   docker build -t pdf-tts .
   docker run -p 8000:8000 pdf-tts
   ```

3. **Access the application**:
   Open your browser to `http://localhost:8000`

### Manual Setup

#### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Download spaCy model**:
   ```bash
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
   ```

3. **Download Supertonic models**:
   ```bash
   git lfs install
   git clone https://huggingface.co/Supertone/supertonic /models/supertonic
   ```

4. **Download helper script**:
   ```bash
   curl -o app/helper.py https://raw.githubusercontent.com/supertone-inc/supertonic/main/py/helper.py
   ```

5. **Run the backend**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

#### Frontend Setup

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Run development server**:
   ```bash
   npm run dev
   ```

3. **Or build for production**:
   ```bash
   npm run build
   ```

## 📖 Usage

1. **Upload a PDF**: Click the "Upload PDF" button and select a PDF file
2. **View Sentences**: The extracted text appears in the viewer, split into sentences
3. **Play Audio**: Click "Play" to start listening from the beginning or current sentence
4. **Navigate**: Use "Next" and "Prev" buttons to skip between sentences
5. **Jump to Sentence**: Double-click any sentence to start playing from that point
6. **Auto-Scroll**: Toggle auto-scroll to automatically follow along with the audio
7. **Playback Controls**: Use Play, Pause, Resume, and Stop buttons as needed

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **PDFMiner.six**: Robust PDF text extraction
- **spaCy**: Industrial-strength NLP for sentence segmentation
- **Supertonic**: State-of-the-art neural TTS with ONNX runtime
- **ONNX Runtime**: Efficient model inference
- **Uvicorn**: Lightning-fast ASGI server

### Frontend
- **Next.js 14**: React framework with static export capabilities
- **React 18**: Component-based UI library
- **Material-UI (MUI) 6**: Comprehensive component library
- **TypeScript**: Type-safe JavaScript development
- **Emotion**: CSS-in-JS styling solution

## 📁 Project Structure

```
document-reader-supertonic/
├── Dockerfile              # Multi-stage Docker build
├── backend/
│   ├── requirements.txt    # Python dependencies
│   └── app/
│       ├── main.py         # FastAPI application & routes
│       ├── pdf_parser.py   # PDF text extraction
│       ├── nlp.py          # Sentence segmentation
│       └── tts.py          # Text-to-speech synthesis
└── frontend/
    ├── package.json        # Node.js dependencies
    ├── next.config.js      # Next.js configuration
    └── src/
        ├── components/     # React components
        │   ├── PdfUploader.tsx     # File upload component
        │   ├── PlayerControls.tsx  # Audio player controls
        │   └── TextViewer.tsx      # Sentence display & navigation
        ├── lib/
        │   └── api.ts      # API client functions
        └── pages/
            └── index.tsx   # Main application page
```

## 🔧 Configuration

### Voice Styles

The default voice style is set to `M1.json` (male voice). You can modify the voice style in `backend/app/tts.py`:

```python
VOICE_STYLE = ["/models/supertonic/voice_styles/M1.json"]
```

Available voice styles are located in `/models/supertonic/voice_styles/`.

### TTS Parameters

Adjust synthesis parameters in the `SupertonicTTS.synthesize()` method:

- `total_step`: Number of diffusion steps (default: 5)
- `speed`: Playback speed multiplier (default: 1.05)

## 🐳 Docker Details

The Dockerfile uses a multi-stage build:

1. **Frontend Stage**: Builds the Next.js application into static files
2. **Backend Stage**: Sets up Python environment, installs dependencies, downloads models, and serves both API and static frontend

## 📝 API Reference

### POST /api/upload

Upload a PDF file and extract sentences.

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "sentences": [
    { "id": 0, "text": "First sentence." },
    { "id": 1, "text": "Second sentence." }
  ]
}
```

### POST /api/tts

Synthesize speech for a given text.

**Request**:
```json
{
  "text": "Text to synthesize"
}
```

**Response**:
```json
{
  "audioUrl": "/data/audio/tmp_xyz.wav"
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project uses the following third-party technologies:
- [Supertonic](https://github.com/supertone-inc/supertonic) - Text-to-speech model
- [spaCy](https://spacy.io/) - Natural language processing
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Next.js](https://nextjs.org/) - React framework

## 🙏 Acknowledgments

- **Supertone** for providing the high-quality TTS models
- **spaCy** for robust NLP capabilities
- The open-source community for all the amazing tools and libraries

## 📧 Contact

For questions, issues, or suggestions, please open an issue on the repository.
