import React, { useState } from "react";
import { Container, Stack, Typography, Box, Button } from "@mui/material";
import PdfUploader from "../components/PdfUploader";
import TextViewer from "../components/TextViewer";
import PlayerControls from "../components/PlayerControls";

type Sentence = { id: number; text: string };

export default function Home() {
  const [sentences, setSentences] = useState<Sentence[]>([]);
  const [currentId, setCurrentId] = useState<number | null>(null);
  const [playRequestId, setPlayRequestId] = useState<number | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  return (
    <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <Container sx={{ py: 2 }}>
        <Typography variant="h4" gutterBottom>
          PDF to Speech
        </Typography>
        <Stack direction="row" spacing={2} alignItems="center">
          <PdfUploader onUploaded={setSentences} />
          <Button
            variant="outlined"
            onClick={() => setAutoScroll(!autoScroll)}
          >
            {autoScroll ? "Disable Auto‑Scroll" : "Enable Auto‑Scroll"}
          </Button>
        </Stack>
      </Container>

      {sentences.length > 0 && (
        <>
          <Box sx={{ flex: 1, overflow: "hidden", px: 2 }}>
            <TextViewer
              sentences={sentences}
              currentId={currentId}
              onJump={(id) => {
                setCurrentId(id);
                setPlayRequestId(id);
              }}
              autoScroll={autoScroll}
            />
          </Box>

          <Box sx={{ p: 2 }}>
            <PlayerControls
              sentences={sentences}
              currentId={currentId}
              onCurrentChange={(id) => {
                setCurrentId(id);
                setPlayRequestId(null);
              }}
              playRequestId={playRequestId}
            />
          </Box>
        </>
      )}
    </Box>
  );
}
