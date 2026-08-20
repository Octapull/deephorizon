import type { InferenceResult, ModelInfo } from "@/lib/types";

export const models: ModelInfo[] = [
  {
    id: "restormer-v1",
    name: "Restormer v1",
    architecture: "Restormer",
    version: "1.0.0",
    description: "Long-range attention for high-fidelity denoising and super-resolution.",
    stage: "Production",
    metrics: { psnr: 33.2, ssim: 0.92, lpips: 0.08 },
  },
  {
    id: "esrgan-v2",
    name: "ESRGAN v2",
    architecture: "ESRGAN",
    version: "2.1.0",
    description: "Perceptual enhancement optimized for sharper ring structures.",
    stage: "Staging",
    metrics: { psnr: 31.8, ssim: 0.89, lpips: 0.1 },
  },
  {
    id: "unet-baseline",
    name: "U-Net Baseline",
    architecture: "U-Net",
    version: "0.9.0",
    description: "Stable baseline for fast reconstruction and model comparison.",
    stage: "Baseline",
    metrics: { psnr: 28.6, ssim: 0.82, lpips: 0.17 },
  },
];

export const sampleResult: InferenceResult = {
  jobId: "demo-7e4c2",
  modelId: "restormer-v1",
  status: "completed",
  imageUrl: "/sample-output.svg",
  metrics: {
    psnr: 33.2,
    ssim: 0.92,
    lpips: 0.08,
    inferenceTimeMs: 312,
  },
};
