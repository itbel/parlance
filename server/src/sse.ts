import type { Response } from "express";
export function openSSE(res: Response) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no"
  });
  res.write("\n");
}
export function sendSSE(res: Response, event: string, data: unknown) {
  const payload = typeof data === "string" ? data : JSON.stringify(data);
  res.write(`event: ${event}\n`);
  res.write(`data: ${payload}\n\n`);
}
export function closeSSE(res: Response) { res.end(); }
