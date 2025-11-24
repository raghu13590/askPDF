import React, { useEffect, useRef } from "react";
import { Box } from "@mui/material";

type Sentence = { id: number; text: string };
type Props = {
  sentences: Sentence[];
  currentId: number | null;
  onJump: (id: number) => void;
  autoScroll: boolean;
};

export default function TextViewer({ sentences, currentId, onJump, autoScroll }: Props) {
  const refs = useRef<(HTMLParagraphElement | null)[]>([]);

  useEffect(() => {
    if (autoScroll && currentId !== null && refs.current[currentId]) {
      refs.current[currentId]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [currentId, autoScroll]);

  return (
    <Box sx={{ border: "1px solid #ccc", borderRadius: 2, p: 2, height: "100%", overflow: "auto" }}>
      {sentences.map((s, idx) => (
        <p
          key={s.id}
          ref={(el) => { refs.current[idx] = el; }}
          style={{
            background: s.id === currentId ? "yellow" : "transparent",
            cursor: "pointer",
            margin: "4px 0",
            transition: "background-color 0.2s ease",
          }}
          onDoubleClick={() => onJump(s.id)}
        >
          {s.text}
        </p>
      ))}
    </Box>
  );
}