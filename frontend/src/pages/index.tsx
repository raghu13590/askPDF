import VerticalAlignCenterIcon from '@mui/icons-material/VerticalAlignCenter';
import React, { useState, useEffect, useRef, useCallback } from "react";
import { Container, Stack, Typography, Box, Button, FormControl, InputLabel, Select, MenuItem, CssBaseline, IconButton, Tooltip } from "@mui/material";
import FluorescentIcon from '@mui/icons-material/Fluorescent';

declare const process: {
  env: Record<string, string | undefined>;
};
import PdfUploader from "../components/PdfUploader";
import PdfViewer from "../components/PdfViewer";
import PlayerControls from "../components/PlayerControls";
import ChatInterface from "../components/ChatInterface";

type Sentence = { id: number; text: string; bboxes: any[] };

export default function Home() {
  const [pdfSentences, setPdfSentences] = useState<Sentence[]>([]);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [activeSource, setActiveSource] = useState<'pdf' | 'chat'>('pdf');
  const [currentPdfId, setCurrentPdfId] = useState<number | null>(null);
  const [currentChatId, setCurrentChatId] = useState<number | null>(null);
  const [playRequestId, setPlayRequestId] = useState<number | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [chatSentences, setChatSentences] = useState<any[]>([]);

  // Highlight toggle
  const [highlightEnabled, setHighlightEnabled] = useState(true);

  const [fileHash, setFileHash] = useState<string | null>(null);

  // Shared embedding model for both upload and chat
  const [embedModel, setEmbedModel] = useState("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isEmbedModelValid, setIsEmbedModelValid] = useState<boolean | null>(null);


  // Resizable chat panel
  const [chatWidth, setChatWidth] = useState(400);
  const [isChatOpen, setIsChatOpen] = useState(true);
  const [isResizing, setIsResizing] = useState(false);
  const chatWidthRef = useRef(400); // Track width during drag without re-renders
  const rafIdRef = useRef<number | null>(null);

  // Fetch available models and order: embedding models first, then the rest (no duplicates)
  useEffect(() => {
  const ragApiUrl = process.env.NEXT_PUBLIC_RAG_API_URL || "http://localhost:8001";
  fetch(`${ragApiUrl}/models`)
    .then(res => res.json())
    .then(data => {
      if (data.embedding_models || data.not_embedding_models) {
        // Show embedding models first, then the rest
        setAvailableModels([...data.embedding_models, ...data.not_embedding_models]);
      } else if (data.all_models) {
        // display all models if specific categories not provided
        setAvailableModels(data.all_models);
      }
    })
    .catch(err => {
      console.warn("Failed to fetch models", err);
    });
}, []);

  // Validate embedding model when changed
  const handleEmbedModelChange = async (embedModel: string) => {
    setEmbedModel(embedModel);
    setIsEmbedModelValid(null); // reset
    if (!embedModel) return;
    try {
      const apiBase = process.env.NEXT_PUBLIC_RAG_API_URL || "http://localhost:8001";
      const res = await fetch(`${apiBase}/health/is_embed_model_ready?model=${encodeURIComponent(embedModel)}`);
      const data = await res.json();
      setIsEmbedModelValid(data.embed_model_ready === true);
      // If a PDF is already loaded, re-trigger upload/indexing to create new collection
      if (fileHash && pdfUrl && data.embed_model_ready === true) {
        // Simulate re-upload by calling the backend index endpoint
        // You may need to call your backend API here to re-index
        const backendApi = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        // Assuming you have an endpoint like /index_document
        // You may need to pass upload_id, file_hash, and embedModel
        try {
          const uploadId = fileHash; // assuming fileHash is used as upload_id
          const indexRes = await fetch(`${apiBase}/index_document`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              text: "", // backend will parse PDF
              embedding_model_name: embedModel,
              metadata: { file_hash: fileHash, upload_id: uploadId }
            })
          });
          const indexData = await indexRes.json();
          // Optionally, handle indexData (status, errors, etc.)
        } catch (err) {
          console.warn("Failed to re-index PDF with new embedding model", err);
        }
      }
    } catch (err) {
      setIsEmbedModelValid(false);
    }
  };

  // Handle resize with optimized performance
  const handleMouseDown = useCallback(() => {
    setIsResizing(true);
    chatWidthRef.current = chatWidth;
  }, [chatWidth]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
    }

    rafIdRef.current = requestAnimationFrame(() => {
      const newWidth = window.innerWidth - e.clientX;
      // Use proportions: min 20%, max 80% of window width
      const minWidth = window.innerWidth * 0.2;
      const maxWidth = window.innerWidth * 0.8;
      const constrainedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
      chatWidthRef.current = constrainedWidth;

      // Update CSS custom property for smooth visual update without re-render
      document.documentElement.style.setProperty('--chat-width', `${constrainedWidth}px`);
    });
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
    // Only update state once when drag ends
    setChatWidth(chatWidthRef.current);
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
    }
  }, []);

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [isResizing, handleMouseMove, handleMouseUp]);

  // Clean up object URL if we were using one (not needed here as we use server URL)

  const canDisplayChat = pdfSentences.length > 0 && !!pdfUrl;
  const chatPanelWidth = isChatOpen
    ? (isResizing ? 'var(--chat-width, 400px)' : chatWidth)
    : 0;

  return (
    <>

      <CssBaseline />
      <Box sx={{ height: "100vh", display: "flex", flexDirection: "row", overflow: "hidden", bgcolor: 'background.default' }}>
        {/* Left Column: PDF Content & Controls */}
        <Box sx={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, borderRight: 1, borderColor: 'divider' }}>
          {/* Top Controls: All in one bar, including PlayerControls */}
          <Box sx={{ px: 2, py: 1, borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
            <Stack direction="row" spacing={2} alignItems="center" justifyContent="center" flexWrap="wrap" useFlexGap>
              <FormControl size="small" sx={{ minWidth: 150 }}>
                <InputLabel id="embed-model-label">Embedding Model</InputLabel>
                <Select
                  labelId="embed-model-label"
                  value={embedModel}
                  label="Embedding Model"
                  displayEmpty
                  onChange={(e) => handleEmbedModelChange(e.target.value)}
                  renderValue={(selected) => selected ? selected : <span style={{ color: '#888' }}>Embedding Model</span>}
                >
                  <MenuItem value="" disabled>
                    <span style={{ color: '#888' }}>Embedding Model</span>
                  </MenuItem>
                  {availableModels.map(m => (
                    <MenuItem key={m} value={m}>{m}</MenuItem>
                  ))}
                </Select>
                {isEmbedModelValid === false && (
                  <Typography color="error" variant="caption" sx={{ ml: 2 }}>
                    Selected model is not a valid embedding model.
                  </Typography>
                )}
              </FormControl>
              <PdfUploader
                embedModel={embedModel}
                onUploaded={(data) => {
                  setPdfSentences(data?.sentences || []);
                  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                  if (data?.pdfUrl) {
                    setPdfUrl(`${apiBase}${data.pdfUrl}?t=${Date.now()}`);
                  }
                  setFileHash(data?.fileHash || null);
                  setCurrentPdfId(null);
                  setCurrentChatId(null);
                  setPlayRequestId(null);
                  setActiveSource('pdf');
                }}
              />
              <Tooltip title={autoScroll ? "Disable Auto-Scroll" : "Enable Auto-Scroll"}>
                <IconButton
                  color={autoScroll ? "primary" : "default"}
                  onClick={() => setAutoScroll(a => !a)}
                  sx={{ border: autoScroll ? 1 : 0, borderColor: autoScroll ? 'primary.main' : 'transparent' }}
                  size="small"
                >
                  <VerticalAlignCenterIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title={highlightEnabled ? "Disable Highlighting" : "Enable Highlighting"}>
                <IconButton
                  color={highlightEnabled ? "primary" : "default"}
                  onClick={() => setHighlightEnabled(h => !h)}
                  sx={{ border: highlightEnabled ? 1 : 0, borderColor: highlightEnabled ? 'primary.main' : 'transparent' }}
                  size="small"
                >
                  <FluorescentIcon />
                </IconButton>
              </Tooltip>
              <Button
                variant="outlined"
                size="small"
                disabled={!canDisplayChat}
                onClick={() => setIsChatOpen(open => !open)}
              >
                {isChatOpen ? "Close Chat" : "Open Chat"}
              </Button>
              {pdfSentences.length > 0 && pdfUrl && (
                <PlayerControls
                  sentences={activeSource === 'pdf' ? pdfSentences : chatSentences}
                  currentId={activeSource === 'pdf' ? currentPdfId : currentChatId}
                  onCurrentChange={(id) => {
                    if (activeSource === 'pdf') {
                      setCurrentPdfId(id);
                    } else {
                      setCurrentChatId(id);
                    }
                    setPlayRequestId(null);
                  }}
                  playRequestId={playRequestId}
                />
              )}
            </Stack>
          </Box>

          <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
            {(pdfSentences?.length ?? 0) > 0 && pdfUrl ? (
              <PdfViewer
                pdfUrl={pdfUrl}
                sentences={pdfSentences}
                currentId={activeSource === 'pdf' ? currentPdfId : null}
                onJump={(id) => {
                  setActiveSource('pdf');
                  setCurrentPdfId(id);
                  setPlayRequestId(id);
                }}
                autoScroll={autoScroll}
                isResizing={isResizing}
                highlightEnabled={highlightEnabled}
              />
            ) : (
              <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'grey.50', p: 4 }}>
                <Box>
                  <Typography variant="h5" color="textSecondary" gutterBottom>
                    Welcome to AskPDF
                  </Typography>
                  <Typography color="textSecondary" sx={{ mb: 2 }}>
                    To get started:
                  </Typography>
                  <ul style={{ color: '#888', margin: 0, paddingLeft: 20, fontSize: 16 }}>
                    <li>Choose an <b>Embedding Model</b> from the dropdown above (required for semantic search and chat).</li>
                    <li>Click <b>Upload PDF</b> and select your document.</li>
                    <li>Wait a moment while your PDF is processed and its content is displayed here.</li>
                    <li>Double-click any text in the PDF or any chat bubble to play audio from that point.</li>
                    <li>Use the <b>Chat</b> panel to ask questions about your PDF using AI.</li>
                    <li>In the chat window, pick a <b>Chat Model</b> from the dropdown before starting your conversation.</li>
                  </ul>
                  <Typography color="textSecondary" sx={{ mt: 2, fontSize: 14 }}>
                    <b>Tip:</b> You can also use the playback controls at the top to play, pause, or skip sentences.
                  </Typography>
                </Box>
              </Box>
            )}
          </Box>
        </Box>

        {/* Resizable Divider */}
        {canDisplayChat && isChatOpen && (
          <Box
            onMouseDown={handleMouseDown}
            sx={{
              width: '12px',
              mx: '-6px',
              cursor: 'col-resize',
              position: 'relative',
              zIndex: 10,
              display: 'flex',
              justifyContent: 'center',
              '&:hover .divider-line, &:active .divider-line': {
                backgroundColor: 'primary.main',
                width: '4px',
              },
            }}
          >
            <Box className="divider-line" sx={{
              width: '2px',
              height: '100%',
              backgroundColor: isResizing ? 'primary.main' : 'divider',
              transition: 'all 0.2s',
            }} />
          </Box>
        )}

        {/* Right Column: Chat Interface */}
        {canDisplayChat && (
          <Box sx={{
            width: chatPanelWidth,
            minWidth: 0,
            height: '100%',
            transition: isResizing || !isChatOpen ? 'none' : 'width 0.1s ease-out',
            bgcolor: 'background.paper',
            visibility: isChatOpen ? 'visible' : 'hidden',
            pointerEvents: isChatOpen ? 'auto' : 'none',
            overflow: 'hidden'
          }}>
            <ChatInterface
              embedModel={embedModel}
              fileHash={fileHash}
              chatSentences={chatSentences}
              setChatSentences={setChatSentences}
              currentChatId={currentChatId}
              activeSource={activeSource}
              onJump={(id) => {
                setActiveSource('chat');
                setCurrentChatId(id);
                setPlayRequestId(id);
              }}
            />
          </Box>
        )}

        {/* Global Drag Mask */}
        {isResizing && (
          <Box sx={{
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            cursor: 'col-resize',
            userSelect: 'none',
            backgroundColor: 'transparent',
          }} />
        )}
      </Box>
    </>
  );
}
