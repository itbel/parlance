import { useRef, useState, useCallback } from "react";

export function useAudioQueue() {
  const queue = useRef<HTMLAudioElement[]>([]);
  const currentAudio = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);

  const playNext = useCallback(() => {
    const next = queue.current.shift();
    if (!next) {
      currentAudio.current = null;
      setPlaying(false);
      return;
    }
    currentAudio.current = next;
    setPlaying(true);
    next.onended = () => {
      currentAudio.current = null;
      playNext();
    };
    next.play().catch(() => {
      currentAudio.current = null;
      playNext();
    });
  }, []);

  const enqueue = useCallback(
    (audio: HTMLAudioElement) => {
      queue.current.push(audio);
      if (!playing) playNext();
    },
    [playing, playNext]
  );

  const stop = useCallback(() => {
    if (currentAudio.current) {
      try {
        currentAudio.current.pause();
        currentAudio.current.currentTime = 0;
      } catch {
        // ignore playback reset errors
      }
      currentAudio.current = null;
    }
    if (queue.current.length) {
      queue.current.forEach((audio) => {
        try {
          audio.pause();
          audio.currentTime = 0;
        } catch {
          // ignore queued playback reset errors
        }
      });
      queue.current = [];
    }
    setPlaying(false);
  }, []);

  return { enqueue, playing, currentAudio, stop };
}
