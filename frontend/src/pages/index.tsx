import React, { useState, useEffect, useRef, useCallback } from "react";
import { Container, Stack, Typography, Box, Button, FormControl, InputLabel, Select, MenuItem, CssBaseline } from "@mui/material";
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

  // Shared embedding model for both upload and chat
  const [embedModel, setEmbedModel] = useState("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  // Resizable chat panel
  const [chatWidth, setChatWidth] = useState(400);
  const [isResizing, setIsResizing] = useState(false);
  const chatWidthRef = useRef(400); // Track width during drag without re-renders
  const rafIdRef = useRef<number | null>(null);

  // Fetch available models
  useEffect(() => {
    const ragApiUrl = process.env.NEXT_PUBLIC_RAG_API_URL || "http://localhost:8001";
    fetch(`${ragApiUrl}/models`)
      .then(res => res.json())
      .then(data => {
        if (data.data) {
          const ids = data.data.map((m: any) => m.id);
          setAvailableModels(ids);
        }
      })
      .catch(err => {
        console.warn("Failed to fetch models", err);
      });
  }, []);

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
      // Constrain width between 300px and 800px
      const constrainedWidth = Math.max(300, Math.min(800, newWidth));
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

  return (
    <>
      <CssBaseline />
      <Box sx={{ height: "100vh", display: "flex", flexDirection: "row", overflow: "hidden", bgcolor: 'background.default' }}>
        {/* Left Column: PDF Content & Controls */}
        <Box sx={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, borderRight: 1, borderColor: 'divider' }}>
          <Box sx={{ px: 2, py: 1, borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
            <Stack direction="row" spacing={2} alignItems="center">
              <Typography variant="h6" noWrap sx={{ fontWeight: 'bold' }}>
                AskPDF
              </Typography>

              <FormControl size="small" sx={{ minWidth: 150 }}>
                <InputLabel>Embedding Model</InputLabel>
                <Select
                  value={embedModel}
                  label="Embedding Model"
                  onChange={(e) => setEmbedModel(e.target.value)}
                >
                  {availableModels.map(m => (
                    <MenuItem key={m} value={m}>{m}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <PdfUploader
                embedModel={embedModel}
                onUploaded={(data) => {
                  setPdfSentences(data.sentences);
                  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                  setPdfUrl(`${apiBase}${data.pdfUrl}?t=${Date.now()}`);
                  setCurrentPdfId(null);
                  setCurrentChatId(null);
                  setPlayRequestId(null);
                  setActiveSource('pdf');
                }}
              />

              <Button
                variant="outlined"
                size="small"
                onClick={() => setAutoScroll(!autoScroll)}
              >
                {autoScroll ? "Disable Auto‑Scroll" : "Enable Auto‑Scroll"}
              </Button>
            </Stack>
          </Box>

          <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
            {pdfSentences.length > 0 && pdfUrl ? (
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
              />
            ) : (
              <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'grey.50' }}>
                <Typography color="textSecondary">Upload a PDF to begin reading</Typography>
              </Box>
            )}
          </Box>

          {pdfSentences.length > 0 && pdfUrl && (
            <Box sx={{ px: 2, py: 1.5, borderTop: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
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
            </Box>
          )}
        </Box>

        {/* Resizable Divider */}
        {pdfSentences.length > 0 && pdfUrl && (
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
        {pdfSentences.length > 0 && pdfUrl && (
          <Box sx={{
            width: isResizing ? 'var(--chat-width, 400px)' : chatWidth,
            height: '100%',
            transition: isResizing ? 'none' : 'width 0.1s ease-out',
            bgcolor: 'background.paper'
          }}>
            <ChatInterface
              embedModel={embedModel}
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
