import { randomUUID } from "crypto";
import { performance } from "node:perf_hooks";
import { recordTimingLog } from "./store.js";

type MetadataValue = string | number | boolean | null | undefined;
type Metadata = Record<string, MetadataValue>;

type TimingStep = {
  name: string;
  durationMs: number;
};

export type TimingStatus = "ok" | "error";

export function createTimingRecorder(event: string, baseMetadata?: Metadata) {
  const requestId = randomUUID();
  const steps: TimingStep[] = [];
  const startWall = Date.now();
  const startPerf = performance.now();
  let finished = false;

  function toMs(duration: number) {
    return Math.round(duration);
  }

  function sanitizeMetadata(meta?: Metadata) {
    if (!meta) return undefined;
    const cleaned: Record<string, string | number | boolean | null> = {};
    for (const [key, value] of Object.entries(meta)) {
      if (value === undefined) continue;
      if (
        typeof value === "string" ||
        typeof value === "number" ||
        typeof value === "boolean" ||
        value === null
      ) {
        cleaned[key] = value;
      } else if (typeof value === "object") {
        cleaned[key] = JSON.stringify(value);
      } else {
        cleaned[key] = String(value);
      }
    }
    return Object.keys(cleaned).length ? cleaned : undefined;
  }

  async function finish(status: TimingStatus, extraMeta?: Metadata) {
    if (finished) return;
    finished = true;
    try {
      await recordTimingLog({
        event,
        status,
        requestId,
        startedAt: new Date(startWall).toISOString(),
        finishedAt: new Date().toISOString(),
        durationMs: toMs(performance.now() - startPerf),
        steps,
        metadata: sanitizeMetadata({ ...baseMetadata, ...extraMeta }),
      });
    } catch (err) {
      console.warn("[telemetry] failed to record timing log", err);
    }
  }

  return {
    requestId,
    async time<T>(name: string, fn: () => Promise<T>): Promise<T> {
      const stepStart = performance.now();
      try {
        return await fn();
      } finally {
        steps.push({ name, durationMs: toMs(performance.now() - stepStart) });
      }
    },
    addStep(name: string, durationMs: number) {
      steps.push({ name, durationMs: toMs(durationMs) });
    },
    finish,
  };
}
