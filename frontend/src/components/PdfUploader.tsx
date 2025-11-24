import { Button } from "@mui/material";
import React, { useRef } from "react";

type Props = { onUploaded: (sentences: { id: number; text: string }[]) => void };

export default function PdfUploader({ onUploaded }: Props) {
  const inputId = "pdf-upload-input";

  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    onUploaded(data.sentences);
    e.target.value = ""; // reset
  };

  return (
    <>
      <input
        id={inputId}
        type="file"
        accept="application/pdf"
        onChange={handleChange}
        style={{ display: "none" }}
      />
      <label htmlFor={inputId}>
        <Button variant="contained" component="span">
          Upload PDF
        </Button>
      </label>
    </>
  );
}
