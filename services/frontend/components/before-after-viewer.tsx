"use client";

import { Columns2, MoveHorizontal } from "lucide-react";
import Image from "next/image";
import { useState } from "react";
import { cn } from "@/lib/utils";

type BeforeAfterViewerProps = {
  beforeUrl: string;
  afterUrl: string;
  className?: string;
  priority?: boolean;
};

export function BeforeAfterViewer({ beforeUrl, afterUrl, className, priority }: BeforeAfterViewerProps) {
  const [position, setPosition] = useState(48);

  return (
    <div className={cn("group relative overflow-hidden rounded-2xl bg-black", className)}>
      <Image src={afterUrl} alt="Enhanced black hole reconstruction" fill className="object-cover" priority={priority} />
      <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}>
        <Image src={beforeUrl} alt="Degraded radio telescope observation" fill className="object-cover" priority={priority} />
      </div>

      <div className="absolute inset-y-0 w-px bg-white/90 shadow-[0_0_16px_rgba(255,255,255,.4)]" style={{ left: `${position}%` }}>
        <span className="absolute left-1/2 top-1/2 grid size-10 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-white/40 bg-black/70 shadow-xl backdrop-blur">
          <MoveHorizontal className="size-4 text-white" />
        </span>
      </div>

      <input
        type="range"
        min="0"
        max="100"
        value={position}
        onChange={(event) => setPosition(Number(event.target.value))}
        className="absolute inset-0 z-10 size-full cursor-ew-resize opacity-0"
        aria-label="Compare original and enhanced images"
      />

      <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-between p-4">
        <span className="rounded-full border border-white/10 bg-black/60 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-200 backdrop-blur">
          Raw observation
        </span>
        <span className="rounded-full border border-ember/20 bg-ember/10 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-orange-100 backdrop-blur">
          AI reconstruction
        </span>
      </div>

      <div className="pointer-events-none absolute bottom-4 right-4 flex items-center gap-2 rounded-full border border-white/10 bg-black/60 px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-400 backdrop-blur">
        <Columns2 className="size-3" /> Drag to compare
      </div>
    </div>
  );
}
