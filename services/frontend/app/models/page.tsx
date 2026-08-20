import { ArrowUpRight, Box, Check, Cpu, Gauge, Orbit } from "lucide-react";
import Link from "next/link";
import { models } from "@/lib/mock-data";
import { cn } from "@/lib/utils";

export default function ModelsPage() {
  return (
    <main className="min-h-screen pt-16">
      <div className="grid-surface pointer-events-none absolute inset-x-0 top-0 h-[640px]" />
      <div className="relative mx-auto max-w-[1480px] px-5 py-16 lg:px-8 lg:py-24">
        <div className="max-w-2xl">
          <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-ember">
            <Orbit className="size-3.5" /> Model registry
          </div>
          <h1 className="mt-4 text-4xl font-medium tracking-[-0.04em] text-bone sm:text-6xl">Validated reconstruction models.</h1>
          <p className="mt-5 max-w-xl text-base leading-7 text-zinc-500">
            Compare architectures, validation metrics and deployment stages before running an observation.
          </p>
        </div>

        <div className="mt-14 grid gap-4 lg:grid-cols-3">
          {models.map((model) => (
            <article key={model.id} className="panel-border group rounded-2xl bg-panel/70 p-5 transition hover:-translate-y-1 hover:bg-panel">
              <div className="flex items-start justify-between">
                <span className="grid size-11 place-items-center rounded-xl border border-white/10 bg-black/20 text-ember">
                  {model.architecture === "Restormer" ? <Orbit className="size-5" /> : model.architecture === "ESRGAN" ? <Cpu className="size-5" /> : <Box className="size-5" />}
                </span>
                <span className={cn(
                  "rounded-full border px-2.5 py-1 font-mono text-[8px] uppercase tracking-[0.14em]",
                  model.stage === "Production" ? "border-emerald-400/20 bg-emerald-400/5 text-emerald-300" : "border-white/10 bg-white/[0.03] text-zinc-500",
                )}>
                  {model.stage}
                </span>
              </div>

              <p className="mt-7 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-600">{model.architecture} · v{model.version}</p>
              <h2 className="mt-2 text-xl font-medium text-bone">{model.name}</h2>
              <p className="mt-3 min-h-12 text-sm leading-6 text-zinc-500">{model.description}</p>

              <div className="mt-6 grid grid-cols-3 gap-px overflow-hidden rounded-xl border border-white/10 bg-white/10">
                {[
                  ["PSNR", model.metrics.psnr],
                  ["SSIM", model.metrics.ssim],
                  ["LPIPS", model.metrics.lpips],
                ].map(([label, value]) => (
                  <div key={label} className="bg-[#0d0f13] p-3">
                    <p className="font-mono text-[8px] uppercase tracking-wider text-zinc-600">{label}</p>
                    <p className="mt-1 text-sm font-medium text-zinc-200">{value}</p>
                  </div>
                ))}
              </div>

              <div className="mt-5 flex items-center justify-between border-t border-white/10 pt-4">
                <span className="inline-flex items-center gap-1.5 text-[10px] text-zinc-500">
                  <Check className="size-3 text-emerald-400" /> Validation complete
                </span>
                <Link href={`/workspace`} className="grid size-8 place-items-center rounded-full border border-white/10 text-zinc-500 transition group-hover:border-ember/30 group-hover:text-ember" aria-label={`Use ${model.name}`}>
                  <ArrowUpRight className="size-3.5" />
                </Link>
              </div>
            </article>
          ))}
        </div>

        <div className="panel-border mt-5 flex flex-col justify-between gap-5 rounded-2xl bg-gradient-to-r from-ember/[0.08] to-transparent p-6 sm:flex-row sm:items-center">
          <div className="flex items-start gap-4">
            <span className="grid size-10 shrink-0 place-items-center rounded-xl border border-ember/20 bg-ember/10 text-ember"><Gauge className="size-4" /></span>
            <div>
              <h2 className="text-sm font-medium text-bone">Need a side-by-side benchmark?</h2>
              <p className="mt-1 text-xs leading-5 text-zinc-500">Batch evaluation and model comparison will be available after the inference contract is finalized.</p>
            </div>
          </div>
          <Link href="/workspace" className="inline-flex h-10 shrink-0 items-center justify-center rounded-full border border-white/10 px-5 text-xs text-zinc-300 hover:bg-white/5">Open workspace</Link>
        </div>
      </div>
    </main>
  );
}
