const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function uploadPdf(file: File, embeddingModel: string) {
  const form = new FormData();
  form.append("file", file);
  form.append("embedding_model", embeddingModel);
  const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function ttsSentence(text: string, voice: string, speed: number) {
  const res = await fetch(`${API_BASE}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice, speed })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
