import React, { useEffect, useRef, useState } from "react";
import { Button, Stack, Select, MenuItem, Slider, Typography, FormControl, InputLabel, IconButton } from "@mui/material";
import { PlayArrow, Pause, SkipPrevious, SkipNext } from '@mui/icons-material';

// For Next.js/browser env
declare const process: {
  env: Record<string, string | undefined>;
};
import { ttsSentence, getVoices } from "../lib/tts-api";

type Sentence = { id: number; text: string };


type Props = {
  sentences: Sentence[];
  currentId: number | null;                // highlight only
  onCurrentChange: (id: number | null) => void;
  playRequestId: number | null;            // explicit command to play now
};

export default function PlayerControls({ sentences, currentId, onCurrentChange, playRequestId }: Props) {
  // Ref to the audio element for playback control
  const audioRef = useRef<HTMLAudioElement>(null);
  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  // Available TTS voices
  const [voices, setVoices] = useState<string[]>([]);
  // Currently selected TTS voice
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  // Playback speed
  const [speed, setSpeed] = useState<number>(1.0);
  // Track paused position for resume
  const [pausedAt, setPausedAt] = useState<number | null>(null);

  // Fetch available TTS voices on mount
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

  // Play a sentence when an external play request is received (e.g., double-click in PDF)
  useEffect(() => {
    if (playRequestId == null) return;
    void playSentence(playRequestId);
  }, [playRequestId]);

  // Restart playback with new voice if changed during playback
  useEffect(() => {
    if (isPlaying && currentId !== null && selectedVoice !== "") {
      void playSentence(currentId);
    }
  }, [selectedVoice]);

  // Stop playback and reset state when sentences change (e.g., new PDF uploaded)
  useEffect(() => {
    // If we're jumping to a new source, the playRequestId effect will handle it.
    // Only auto-stop for passive changes
    if (playRequestId !== null) return;

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current.onended = null;
    }
    setIsPlaying(false);
    setPausedAt(null);
  }, [sentences]);

  /**
   * Play the sentence at the given index. If resumeFrom is provided, resumes from that time.
   * Handles TTS audio fetching and playback, and auto-advances to next sentence on end.
   */
  async function playSentence(id: number, resumeFrom?: number) {
    if (selectedVoice === "") {
      console.warn("No voice selected, skipping playback.");
      return;
    }
    const audio = audioRef.current;
    if (!audio) return;

    // Stop any current playback
    audio.pause();
    audio.currentTime = 0;
    audio.onended = null;

    const s = sentences[id];
    onCurrentChange(id);

    try {
      const { audioUrl } = await ttsSentence(s.text, selectedVoice, speed);
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      audio.src = `${apiBase}${audioUrl}`;
      await audio.play();
      if (resumeFrom) {
        audio.currentTime = resumeFrom;
      }
      setIsPlaying(true);
      setPausedAt(null);

      // Auto-advance to next sentence on playback end
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

  /**
   * Toggle play/pause for the current sentence.
   * If paused, resumes from last position; otherwise, starts from current or first sentence.
   */
  function handlePlayPause() {
    const audio = audioRef.current;
    if (!isPlaying) {
      if (pausedAt !== null && currentId !== null) {
        audio!.play();
        setIsPlaying(true);
        setPausedAt(null);
      } else {
        void playSentence(currentId ?? 0);
      }
    } else {
      if (audio) {
        audio.pause();
        setPausedAt(audio.currentTime);
        setIsPlaying(false);
      }
    }
  }

  return (
    <Stack direction="row" spacing={2} alignItems="center" useFlexGap sx={{ flexWrap: "wrap" }}>
      <Stack direction="row" spacing={1} alignItems="center" useFlexGap sx={{ flexWrap: "wrap" }}>
        <IconButton color="primary" onClick={handlePlayPause} size="large">
          {isPlaying ? <Pause fontSize="large" /> : <PlayArrow fontSize="large" />}
        </IconButton>
        <IconButton onClick={() => currentId !== null && currentId > 0 && playSentence(currentId - 1)} disabled={currentId === null || currentId <= 0} size="large">
          <SkipPrevious />
        </IconButton>
        <IconButton onClick={() => currentId !== null && currentId < sentences.length - 1 && playSentence(currentId + 1)} disabled={currentId === null || currentId >= sentences.length - 1} size="large">
          <SkipNext />
        </IconButton>
      </Stack>

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

      <Stack direction="row" spacing={1} alignItems="center" useFlexGap sx={{ flexWrap: "wrap" }}>
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
