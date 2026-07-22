export type JobStatus = "idle" | "ready" | "queued" | "running" | "completed" | "failed";

export type ModelArchitecture = "U-Net" | "Pix2Pix" | "ESRGAN" | "Restormer";

export type ModelInfo = {
  id: string;
  name: string;
  architecture: ModelArchitecture;
  version: string;
  description: string;
  stage: "Production" | "Staging" | "Baseline";
  metrics: {
    psnr: number;
    ssim: number;
    lpips: number;
  };
};

export type InferenceResult = {
  jobId: string;
  modelId: string;
  status: "completed";
  imageUrl: string;
  metrics: {
    psnr: number;
    ssim: number;
    lpips: number;
    inferenceTimeMs: number;
  };
};
