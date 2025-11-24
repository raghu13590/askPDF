import React, { useEffect, useRef, useState } from "react";
import { Button, Stack } from "@mui/material";
import { ttsSentence } from "../lib/api";

type Sentence = { id: number; text: string };

type Props = {
  sentences: Sentence[];
  currentId: number | null;                // highlight only
  onCurrentChange: (id: number | null) => void;
  playRequestId: number | null;            // explicit command to play now
};

export default function PlayerControls({ sentences, currentId, onCurrentChange, playRequestId }: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // React to external play requests (double‑click)
  useEffect(() => {
    if (playRequestId == null) return;
    void playSentence(playRequestId);
  }, [playRequestId]);

  async function playSentence(id: number) {
    const audio = audioRef.current;
    if (!audio) return;

    // Interrupt current playback and clear old handler
    audio.pause();
    audio.currentTime = 0;
    audio.onended = null;

    const s = sentences[id];
    onCurrentChange(id);

    const { audioUrl } = await ttsSentence(s.text);

    audio.src = audioUrl;
    await audio.play();
    setIsPlaying(true);

    // Attach ended handler for this playback
    audio.onended = () => {
      const next = id + 1;
      if (next < sentences.length) {
        void playSentence(next);
      } else {
        setIsPlaying(false);
        onCurrentChange(null);
      }
    };
  }

  function pause() {
    audioRef.current?.pause();
    setIsPlaying(false);
  }

  function resume() {
    audioRef.current?.play();
    setIsPlaying(true);
  }

  function stop() {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current.onended = null;
    }
    setIsPlaying(false);
    onCurrentChange(null);
  }

  return (
    <Stack direction="row" spacing={2} alignItems="center">
      <Button variant="contained" onClick={() => playSentence(currentId ?? 0)}>Play</Button>
      <Button variant="outlined" onClick={pause} disabled={!isPlaying}>Pause</Button>
      <Button variant="outlined" onClick={resume} disabled={isPlaying}>Resume</Button>
      <Button variant="text" onClick={stop}>Stop</Button>
      <Button variant="outlined" onClick={() => currentId !== null && currentId > 0 && playSentence(currentId - 1)}>
        ⏮️ Prev
      </Button>
      <Button variant="outlined" onClick={() => currentId !== null && currentId < sentences.length - 1 && playSentence(currentId + 1)}>
        ⏭️ Next
      </Button>
      <audio ref={audioRef} />
    </Stack>
  );
}
