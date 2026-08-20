"use client";

import { useQuery } from "@tanstack/react-query";
import {
  AlertCircle,
  ArrowDownToLine,
  Check,
  ChevronDown,
  CircleGauge,
  FileImage,
  LoaderCircle,
  Maximize2,
  Orbit,
  RotateCcw,
  Sparkles,
  UploadCloud,
  X,
} from "lucide-react";
import Image from "next/image";
import { useRef, useState, type ChangeEvent } from "react";
import { BeforeAfterViewer } from "@/components/before-after-viewer";
import { listModels, runMockInference } from "@/lib/api";
import type { InferenceResult, JobStatus } from "@/lib/types";
import { cn, formatBytes } from "@/lib/utils";

const scaleOptions = [1, 2, 4] as const;
const formatOptions = ["png", "fits"] as const;

export function Workspace() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("/sample-input.svg");
  const [modelId, setModelId] = useState("restormer-v1");
  const [scaleFactor, setScaleFactor] = useState<(typeof scaleOptions)[number]>(4);
  const [outputFormat, setOutputFormat] = useState<(typeof formatOptions)[number]>("png");
  const [status, setStatus] = useState<JobStatus>("ready");
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const modelQuery = useQuery({ queryKey: ["models"], queryFn: listModels });
  const selectedModel = modelQuery.data?.find((model) => model.id === modelId);

  function selectFile(selectedFile?: File) {
    if (!selectedFile) return;

    const isFits = /\.(fits|fit)$/i.test(selectedFile.name);
    const isImage = selectedFile.type.startsWith("image/");

    if (!isFits && !isImage) {
      setError("Please select a PNG, JPEG or FITS observation.");
      return;
    }

    if (selectedFile.size > 20 * 1024 * 1024) {
      setError("The observation must be smaller than 20 MB.");
      return;
    }

    if (previewUrl.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    setFile(selectedFile);
    setPreviewUrl(isImage ? URL.createObjectURL(selectedFile) : "/sample-input.svg");
    setResult(null);
    setStatus("ready");
    setError(null);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0]);
  }

  function clearFile() {
    if (previewUrl.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl("/sample-input.svg");
    setResult(null);
    setStatus("ready");
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function enhance() {
    setStatus("running");
    setError(null);

    try {
      const inferenceResult = await runMockInference();
      setResult(inferenceResult);
      setStatus("completed");
    } catch {
      setStatus("failed");
      setError("The reconstruction could not be completed. Please try again.");
    }
  }

  return (
    <div className="mx-auto max-w-[1600px] px-4 pb-10 pt-20 sm:px-6 lg:px-8">
      <div className="mb-6 flex flex-col justify-between gap-4 pt-5 sm:flex-row sm:items-end">
        <div>
          <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-ember">
            <Orbit className="size-3.5" /> Reconstruction workspace
          </div>
          <h1 className="mt-2 text-2xl font-medium tracking-tight text-bone sm:text-3xl">Observation analysis</h1>
        </div>
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <span className="size-1.5 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,.8)]" />
          Mock inference environment
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
        <aside className="panel-border h-fit overflow-hidden rounded-2xl bg-panel/80 backdrop-blur xl:sticky xl:top-20">
          <div className="border-b border-white/10 px-5 py-4">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-500">Input controls</p>
          </div>

          <div className="space-y-6 p-5">
            <section>
              <div className="mb-3 flex items-center justify-between">
                <label className="text-sm font-medium text-zinc-200">Observation</label>
                <span className="font-mono text-[9px] uppercase tracking-wider text-zinc-600">PNG · JPG · FITS</span>
              </div>

              {file ? (
                <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-black/20 p-3">
                  <div className="relative size-12 overflow-hidden rounded-lg bg-black">
                    <Image src={previewUrl} alt="Selected observation" fill className="object-cover" unoptimized={previewUrl.startsWith("blob:")} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-zinc-200">{file.name}</p>
                    <p className="mt-1 font-mono text-[9px] uppercase tracking-wider text-zinc-600">{formatBytes(file.size)}</p>
                  </div>
                  <button type="button" onClick={clearFile} className="grid size-8 place-items-center rounded-lg text-zinc-500 hover:bg-white/5 hover:text-white" aria-label="Remove file">
                    <X className="size-4" />
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => {
                    event.preventDefault();
                    selectFile(event.dataTransfer.files[0]);
                  }}
                  className="group flex w-full flex-col items-center rounded-xl border border-dashed border-white/15 bg-white/[0.02] px-4 py-7 text-center transition hover:border-ember/40 hover:bg-ember/[0.03]"
                >
                  <span className="grid size-10 place-items-center rounded-full border border-white/10 bg-white/[0.03] transition group-hover:border-ember/30 group-hover:text-ember">
                    <UploadCloud className="size-4" />
                  </span>
                  <span className="mt-3 text-xs font-medium text-zinc-300">Drop observation or browse</span>
                  <span className="mt-1 text-[10px] text-zinc-600">Maximum file size: 20 MB</span>
                </button>
              )}
              <input ref={inputRef} type="file" accept="image/png,image/jpeg,.fits,.fit" onChange={onFileChange} className="sr-only" />
            </section>

            <section>
              <label htmlFor="model" className="mb-3 block text-sm font-medium text-zinc-200">Reconstruction model</label>
              <div className="relative">
                <select
                  id="model"
                  value={modelId}
                  onChange={(event) => setModelId(event.target.value)}
                  className="h-11 w-full appearance-none rounded-xl border border-white/10 bg-black/25 px-3 pr-9 text-xs text-zinc-200 outline-none transition focus:border-ember/40"
                >
                  {(modelQuery.data ?? []).map((model) => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-zinc-500" />
              </div>
              {selectedModel && (
                <div className="mt-2 flex items-center justify-between font-mono text-[9px] uppercase tracking-wider text-zinc-600">
                  <span>{selectedModel.architecture}</span>
                  <span className="text-emerald-400">{selectedModel.stage}</span>
                </div>
              )}
            </section>

            <section>
              <p className="mb-3 text-sm font-medium text-zinc-200">Scale factor</p>
              <div className="grid grid-cols-3 gap-2">
                {scaleOptions.map((scale) => (
                  <button
                    key={scale}
                    type="button"
                    onClick={() => setScaleFactor(scale)}
                    className={cn(
                      "h-10 rounded-xl border text-xs transition",
                      scaleFactor === scale
                        ? "border-ember/50 bg-ember/10 font-medium text-orange-100"
                        : "border-white/10 bg-black/20 text-zinc-500 hover:text-zinc-300",
                    )}
                  >
                    {scale}×
                  </button>
                ))}
              </div>
            </section>

            <section>
              <p className="mb-3 text-sm font-medium text-zinc-200">Output format</p>
              <div className="grid grid-cols-2 gap-2">
                {formatOptions.map((format) => (
                  <button
                    key={format}
                    type="button"
                    onClick={() => setOutputFormat(format)}
                    className={cn(
                      "flex h-10 items-center justify-center gap-2 rounded-xl border font-mono text-[10px] uppercase tracking-wider transition",
                      outputFormat === format
                        ? "border-white/20 bg-white/[0.07] text-white"
                        : "border-white/10 bg-black/20 text-zinc-600 hover:text-zinc-300",
                    )}
                  >
                    {outputFormat === format && <Check className="size-3" />}
                    {format}
                  </button>
                ))}
              </div>
            </section>

            {error && (
              <div className="flex gap-2 rounded-xl border border-red-400/20 bg-red-400/5 p-3 text-xs leading-5 text-red-200">
                <AlertCircle className="mt-0.5 size-4 shrink-0" />
                {error}
              </div>
            )}

            <button
              type="button"
              onClick={enhance}
              disabled={status === "running" || modelQuery.isLoading}
              className="group flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-bone text-sm font-semibold text-ink transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
            >
              {status === "running" ? (
                <><LoaderCircle className="size-4 animate-spin" /> Reconstructing</>
              ) : (
                <><Sparkles className="size-4 text-ember" /> Enhance observation</>
              )}
            </button>
          </div>
        </aside>

        <section className="min-w-0 space-y-5">
          <div className="panel-border overflow-hidden rounded-2xl bg-panel/70 p-2">
            <div className="relative aspect-square overflow-hidden rounded-xl bg-black sm:aspect-[16/10] xl:aspect-[16/9]">
              {status === "completed" && result ? (
                <BeforeAfterViewer beforeUrl={previewUrl} afterUrl={result.imageUrl} className="size-full rounded-xl" />
              ) : (
                <>
                  <Image src={previewUrl} alt="Observation preview" fill className="object-contain" unoptimized={previewUrl.startsWith("blob:")} />
                  <div className="absolute inset-x-0 top-0 flex items-center justify-between bg-gradient-to-b from-black/70 to-transparent p-4 sm:p-5">
                    <span className="rounded-full border border-white/10 bg-black/50 px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-300 backdrop-blur">Input observation</span>
                    <button type="button" className="grid size-8 place-items-center rounded-full border border-white/10 bg-black/50 text-zinc-400 backdrop-blur hover:text-white" aria-label="Fullscreen preview">
                      <Maximize2 className="size-3.5" />
                    </button>
                  </div>
                  {status === "running" && (
                    <div className="absolute inset-0 grid place-items-center bg-black/65 backdrop-blur-sm">
                      <div className="text-center">
                        <div className="relative mx-auto grid size-16 place-items-center rounded-full border border-ember/30 bg-ember/10">
                          <LoaderCircle className="size-6 animate-spin text-ember" />
                          <span className="scan-line absolute inset-x-2 top-0 h-px bg-gradient-to-r from-transparent via-orange-200 to-transparent" />
                        </div>
                        <p className="mt-5 text-sm font-medium text-bone">Reconstructing event horizon</p>
                        <p className="mt-2 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500">GPU inference in progress</p>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {status === "completed" && result ? (
            <ResultMetrics result={result} onReset={() => { setResult(null); setStatus("ready"); }} />
          ) : (
            <div className="grid gap-3 sm:grid-cols-3">
              {[
                [FileImage, "Input", file ? file.name : "Synthetic EHT sample"],
                [CircleGauge, "Target scale", `${scaleFactor}× super-resolution`],
                [Sparkles, "Model", selectedModel?.name ?? "Loading models"],
              ].map(([Icon, label, value]) => {
                const CardIcon = Icon as typeof FileImage;
                return (
                  <div key={String(label)} className="panel-border rounded-xl bg-panel/60 p-4">
                    <CardIcon className="size-4 text-ember" />
                    <p className="mt-4 font-mono text-[9px] uppercase tracking-[0.15em] text-zinc-600">{String(label)}</p>
                    <p className="mt-1 truncate text-xs text-zinc-300">{String(value)}</p>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function ResultMetrics({ result, onReset }: { result: InferenceResult; onReset: () => void }) {
  const metrics = [
    { label: "PSNR", value: `${result.metrics.psnr} dB`, target: "Target ≥ 32", pass: result.metrics.psnr >= 32 },
    { label: "SSIM", value: result.metrics.ssim.toFixed(2), target: "Target ≥ 0.90", pass: result.metrics.ssim >= 0.9 },
    { label: "LPIPS", value: result.metrics.lpips.toFixed(2), target: "Target ≤ 0.10", pass: result.metrics.lpips <= 0.1 },
    { label: "Latency", value: `${result.metrics.inferenceTimeMs} ms`, target: "Target ≤ 500", pass: result.metrics.inferenceTimeMs <= 500 },
  ];

  return (
    <div className="panel-border rounded-2xl bg-panel/70 p-5">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <div className="flex items-center gap-2">
            <span className="grid size-5 place-items-center rounded-full bg-emerald-400/10 text-emerald-400"><Check className="size-3" /></span>
            <h2 className="text-sm font-medium text-bone">Reconstruction complete</h2>
          </div>
          <p className="mt-1 pl-7 font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-600">Job {result.jobId}</p>
        </div>
        <div className="flex gap-2">
          <button type="button" onClick={onReset} className="inline-flex h-9 items-center gap-2 rounded-lg border border-white/10 px-3 text-xs text-zinc-400 hover:text-white">
            <RotateCcw className="size-3.5" /> Reset
          </button>
          <a href={result.imageUrl} download className="inline-flex h-9 items-center gap-2 rounded-lg bg-bone px-3 text-xs font-medium text-ink hover:bg-white">
            <ArrowDownToLine className="size-3.5" /> Download
          </a>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-white/10 bg-white/10 lg:grid-cols-4">
        {metrics.map((metric) => (
          <div key={metric.label} className="bg-[#0d0f13] p-4 sm:p-5">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[9px] uppercase tracking-[0.15em] text-zinc-600">{metric.label}</span>
              <span className={cn("size-1.5 rounded-full", metric.pass ? "bg-emerald-400" : "bg-amber-400")} />
            </div>
            <p className="mt-3 text-xl font-medium tracking-tight text-bone">{metric.value}</p>
            <p className="mt-1 text-[10px] text-zinc-600">{metric.target}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
