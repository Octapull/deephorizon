import { models, sampleResult } from "@/lib/mock-data";
import type { InferenceResult, ModelInfo } from "@/lib/types";

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8080";

export async function listModels(): Promise<ModelInfo[]> {
  if (process.env.NEXT_PUBLIC_USE_MOCK_API !== "false") {
    return models;
  }

  const response = await fetch(`${apiBaseUrl}/models`);
  if (!response.ok) throw new Error("Modeller alınamadı.");
  const data = (await response.json()) as { models: ModelInfo[] };
  return data.models;
}

export async function runMockInference(): Promise<InferenceResult> {
  await new Promise((resolve) => setTimeout(resolve, 1800));
  return sampleResult;
}
