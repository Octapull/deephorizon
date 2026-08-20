import { ArrowRight, ChartNoAxesCombined, RadioTower, ScanLine, Sparkles } from "lucide-react";
import Link from "next/link";
import { BeforeAfterViewer } from "@/components/before-after-viewer";

const metrics = [
  { value: "33.2 dB", label: "Peak signal-to-noise", icon: ChartNoAxesCombined },
  { value: "0.92", label: "Structural similarity", icon: ScanLine },
  { value: "312 ms", label: "GPU inference", icon: Sparkles },
];

export default function HomePage() {
  return (
    <main className="relative overflow-hidden pt-16">
      <div className="grid-surface pointer-events-none absolute inset-x-0 top-0 h-[900px]" />
      <div className="absolute left-1/2 top-32 size-[540px] -translate-x-1/2 rounded-full bg-ember/10 blur-[140px]" />

      <section className="relative mx-auto grid min-h-[calc(100vh-4rem)] max-w-[1480px] items-center gap-14 px-5 py-20 lg:grid-cols-[0.8fr_1.2fr] lg:px-8 lg:py-24">
        <div className="relative z-10 max-w-xl">
          <div className="mb-7 inline-flex items-center gap-2 rounded-full border border-ember/20 bg-ember/5 px-3 py-1.5">
            <RadioTower className="size-3.5 text-ember" />
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-orange-200">Event Horizon Intelligence</span>
          </div>

          <h1 className="text-balance text-5xl font-medium leading-[0.98] tracking-[-0.055em] text-bone sm:text-7xl lg:text-[5.8rem]">
            See beyond the <span className="bg-gradient-to-r from-ember via-solar to-yellow-100 bg-clip-text text-transparent">blur.</span>
          </h1>

          <p className="mt-7 max-w-lg text-base leading-7 text-zinc-400 sm:text-lg">
            Reconstruct physically consistent, high-resolution black hole images from noisy radio telescope observations.
          </p>

          <div className="mt-9 flex flex-col gap-3 sm:flex-row">
            <Link
              href="/workspace"
              className="group inline-flex h-12 items-center justify-center gap-2 rounded-full bg-bone px-6 text-sm font-medium text-ink transition hover:bg-white"
            >
              Open workspace
              <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
            </Link>
            <Link
              href="/models"
              className="inline-flex h-12 items-center justify-center rounded-full border border-white/10 bg-white/[0.03] px-6 text-sm text-zinc-300 transition hover:border-white/20 hover:bg-white/[0.06]"
            >
              Explore models
            </Link>
          </div>

          <div className="mt-12 grid grid-cols-3 gap-5 border-t border-white/10 pt-7">
            {metrics.map((metric) => (
              <div key={metric.label}>
                <div className="flex items-center gap-2">
                  <metric.icon className="size-3.5 text-ember" />
                  <span className="font-mono text-sm font-semibold text-bone sm:text-base">{metric.value}</span>
                </div>
                <p className="mt-2 text-[10px] leading-4 text-zinc-500 sm:text-xs">{metric.label}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="relative mx-auto w-full max-w-3xl lg:ml-auto">
          <div className="absolute -inset-12 rounded-full bg-ember/10 blur-3xl" />
          <div className="panel-border relative rounded-[1.6rem] bg-white/[0.035] p-2 shadow-glow backdrop-blur">
            <BeforeAfterViewer
              beforeUrl="/sample-input.svg"
              afterUrl="/sample-output.svg"
              className="aspect-square sm:aspect-[1.18/1]"
              priority
            />
          </div>
          <div className="absolute -bottom-5 left-6 rounded-xl border border-white/10 bg-panel/90 px-4 py-3 shadow-xl backdrop-blur md:left-10">
            <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-500">Active model</p>
            <p className="mt-1 text-sm font-medium text-bone">Restormer v1 · 4× SR</p>
          </div>
        </div>
      </section>

      <section className="relative border-t border-white/10 bg-black/20">
        <div className="mx-auto grid max-w-[1480px] gap-8 px-5 py-16 lg:grid-cols-3 lg:px-8">
          {[
            ["01", "Upload", "Drop a PNG or FITS observation into a secure analysis workspace."],
            ["02", "Reconstruct", "Choose a validated model and run GPU-accelerated enhancement."],
            ["03", "Validate", "Compare imagery and inspect PSNR, SSIM, LPIPS and inference time."],
          ].map(([step, title, text]) => (
            <article key={step} className="border-l border-white/10 pl-5">
              <span className="font-mono text-[10px] tracking-[0.2em] text-ember">{step}</span>
              <h2 className="mt-4 text-xl font-medium text-bone">{title}</h2>
              <p className="mt-3 max-w-sm text-sm leading-6 text-zinc-500">{text}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
