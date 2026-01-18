

"""
PDFService: High-level business logic for PDF upload, extraction, and RAG indexing.

This service handles:
- Saving uploaded PDF files
- Extracting text and character coordinates
- Splitting text into sentences
- Mapping sentences to bounding boxes
- Triggering background RAG indexing

Dependencies:
- fastapi.HTTPException for error handling
- httpx for async HTTP requests
- pdf_parser and nlp modules for PDF/text processing
"""

import os
import uuid
import hashlib
import httpx
import logging
from fastapi import HTTPException
from .pdf_parser import extract_text_with_coordinates
from .nlp import split_into_sentences


class PDFService:
    """
    Service class for handling PDF uploads, extraction, and RAG indexing.
    """
    def __init__(self, static_dir="/static", rag_service_url=None):
        """
        Initialize the PDFService.

        Args:
            static_dir (str): Directory to save uploaded PDFs.
            rag_service_url (str): URL for the RAG service. If not provided, uses RAG_SERVICE_URL env var.
        Raises:
            RuntimeError: If RAG_SERVICE_URL is not set.
        """
        self.static_dir = static_dir
        self.rag_service_url = rag_service_url or os.getenv("RAG_SERVICE_URL")
        if not self.rag_service_url:
            logging.error("RAG_SERVICE_URL is not set. Please set the environment variable or pass it explicitly.")
            raise RuntimeError("RAG_SERVICE_URL is not set. Please set the environment variable or pass it explicitly.")
        os.makedirs(self.static_dir, exist_ok=True)

    async def process_upload(self, file, embedding_model, background_tasks):
        """
        Handle the upload of a PDF file, extract text and bounding boxes, and trigger RAG indexing.

        Args:
            file (UploadFile): The uploaded PDF file.
            embedding_model (str): The embedding model to use for RAG.
            background_tasks (BackgroundTasks): FastAPI background tasks for async RAG call.
        Returns:
            dict: Contains sentences with bounding boxes, PDF URL, and file hash.
        Raises:
            HTTPException: If file is not a PDF or embedding_model is missing.
        """
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")
        if not embedding_model:
            raise HTTPException(status_code=400, detail="Please provide an embedding_model.")

        upload_id = str(uuid.uuid4())
        pdf_filename = f"{upload_id}.pdf"
        pdf_path = os.path.join(self.static_dir, pdf_filename)

        content = await file.read()
        file_hash = hashlib.md5(content).hexdigest()

        with open(pdf_path, "wb") as f:
            f.write(content)

        text, char_map = extract_text_with_coordinates(content)
        sentences = split_into_sentences(text)

        enriched_sentences = self._map_sentences_to_bboxes(sentences, text, char_map)

        # Trigger RAG Indexing in background
        background_tasks.add_task(
            self._call_rag,
            text,
            {"filename": file.filename, "upload_id": upload_id, "file_hash": file_hash},
            embedding_model
        )

        return {
            "sentences": enriched_sentences,
            "pdfUrl": f"/{pdf_filename}",
            "fileHash": file_hash
        }

    @staticmethod
    def _map_sentences_to_bboxes(sentences, text, char_map):
        """
        Map each sentence to its bounding boxes by matching text in the extracted PDF.

        Args:
            sentences (list): List of sentence dicts with 'text' key.
            text (str): The full extracted text from the PDF.
            char_map (list): List of character coordinate dicts.
        Returns:
            list: Sentences with 'bboxes' key added for each.
        """
        current_idx = 0
        enriched_sentences = []
        for s in sentences:
            s_text = s["text"]
            s_text_clean = "".join(s_text.split())
            if not s_text_clean:
                s["bboxes"] = []
                enriched_sentences.append(s)
                continue
            match_start = -1
            match_end = -1
            s_ptr = 0
            temp_idx = current_idx
            while temp_idx < len(text) and s_ptr < len(s_text_clean):
                char = text[temp_idx]
                if char.isspace():
                    temp_idx += 1
                    continue
                if char == s_text_clean[s_ptr]:
                    if match_start == -1:
                        match_start = temp_idx
                    s_ptr += 1
                    temp_idx += 1
                else:
                    if match_start != -1:
                        temp_idx = match_start + 1
                        match_start = -1
                        s_ptr = 0
                    else:
                        temp_idx += 1
            if match_start != -1 and s_ptr == len(s_text_clean):
                match_end = temp_idx
                bboxes = char_map[match_start:match_end]
                s["bboxes"] = bboxes
                current_idx = match_end
            else:
                s["bboxes"] = []
            enriched_sentences.append(s)
        return enriched_sentences

    async def _call_rag(self, txt: str, metadata: dict, emb_model: str):
        """
        Asynchronously call the RAG service to index the extracted text.

        Args:
            txt (str): The extracted text to index.
            metadata (dict): Metadata about the PDF/file.
            emb_model (str): The embedding model to use.
        """
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{self.rag_service_url}/index",
                    json={
                        "text": txt,
                        "embedding_model": emb_model,
                        "metadata": metadata
                    },
                    timeout=300.0
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"RAG Indexing failed: {repr(e)}", flush=True)
