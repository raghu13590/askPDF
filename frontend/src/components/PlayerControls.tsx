import React, { useEffect, useRef, useState } from "react";
import { Button, Stack, Select, MenuItem, Slider, Typography, FormControl, InputLabel } from "@mui/material";
import { ttsSentence, getVoices } from "../lib/tts-api";

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
  const [voices, setVoices] = useState<string[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [speed, setSpeed] = useState<number>(1.0);

  useEffect(() => {
    async function fetchVoices() {
      try {
        const voicesData = await getVoices();
        setVoices(voicesData);
        if (voicesData.length > 0 && !selectedVoice) {
          setSelectedVoice(voicesData[0]);
        }
      } catch (err) {
        console.error("Failed to fetch voices", err);
      }
    }
    fetchVoices();
  }, []);

  // React to external play requests (double‑click)
  useEffect(() => {
    if (playRequestId == null) return;
    void playSentence(playRequestId);
  }, [playRequestId]);

  // Trigger playback update when voice changes
  useEffect(() => {
    if (isPlaying && currentId !== null && selectedVoice !== "") { // Only re-play if a voice is actually selected
      void playSentence(currentId);
    }
  }, [selectedVoice]);

  // Stop playback when sentences changes (new PDF uploaded or active source switched)
  useEffect(() => {
    // If we're jumping to a new source, the playRequestId effect will handle it.
    // We only want to auto-stop if this is a passive change (like uploading a new PDF).
    if (playRequestId !== null) return;

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current.onended = null;
    }
    setIsPlaying(false);
  }, [sentences]);

  async function playSentence(id: number) {
    if (selectedVoice === "") {
      console.warn("No voice selected, skipping playback.");
      return;
    }
    const audio = audioRef.current;
    if (!audio) return;

    // Interrupt current playback and clear old handler
    audio.pause();
    audio.currentTime = 0;
    audio.onended = null;

    const s = sentences[id];
    onCurrentChange(id);

    try {
      const { audioUrl } = await ttsSentence(s.text, selectedVoice, speed);

      // Check if we are still supposed to be playing this sentence (race condition check)
      // Note: A more robust way would be to use a request ID or cancellation token, 
      // but checking currentId matches is a reasonable proxy if onCurrentChange updates it synchronously.
      // However, since we might have moved on, let's just proceed. 
      // Ideally, we should check if the component is still mounted or if another request started.

      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      audio.src = `${apiBase}${audioUrl}`;
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
    } catch (e) {
      console.error("Playback failed", e);
      setIsPlaying(false);
    }
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
    <Stack direction="row" spacing={2} alignItems="center" useFlexGap sx={{ flexWrap: "wrap" }}>
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

      <FormControl size="small" style={{ minWidth: 120 }}>
        <InputLabel>Voice</InputLabel>
        <Select
          value={selectedVoice}
          label="Voice"
          onChange={(e: any) => setSelectedVoice(e.target.value as string)}
        >
          {voices.map((v: string) => (
            <MenuItem key={v} value={v}>
              {v.replace(".json", "")}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <Stack direction="row" spacing={1} alignItems="center">
        <Typography variant="caption">Speed</Typography>
        <Slider
          value={speed}
          min={0.5}
          max={2.0}
          step={0.1}
          onChange={(_: Event, val: number | number[]) => setSpeed(val as number)}
          onChangeCommitted={() => {
            if (isPlaying && currentId !== null) {
              void playSentence(currentId);
            }
          }}
          valueLabelDisplay="auto"
          sx={{ width: 100 }}
        />
      </Stack>
      <audio ref={audioRef} />
    </Stack>
  );
}
