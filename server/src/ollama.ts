type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export async function* streamOllama(ollamaBase: string, model: string, messages: ChatMessage[], options?: { temperature?: number }) {
  const body = { model, messages, stream: true, options: { temperature: options?.temperature ?? 0.7 } };
  const resp = await fetch(`${ollamaBase.replace(/\/+$/, "")}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!resp.ok || !resp.body) throw new Error(`Ollama error: ${resp.status} ${resp.statusText}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      try {
        const obj = JSON.parse(line);
        const token = obj.message?.content ?? "";
        const doneFlag = obj.done === true;
        yield { token, done: doneFlag };
      } catch {}
    }
  }
}
