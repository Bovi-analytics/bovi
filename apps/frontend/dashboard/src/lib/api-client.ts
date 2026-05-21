import type { z } from "zod";
import { getApiBaseUrl } from "@/lib/env";
import {
  AutoencoderPredictResponseSchema,
  CharacteristicResponseSchema,
  FitResponseSchema,
  HerdProfileListSchema,
  HerdProfileSchema,
  HerdProfileUploadResponseSchema,
  PresetDatasetResponseSchema,
  PresetHerdStatsResponseSchema,
  PredictResponseSchema,
  TestIntervalResponseSchema,
  YieldEstimateResponseSchema,
  ChallengeListSchema,
  ChallengeReadSchema,
  SubmissionListSchema,
  SubmissionReadSchema,
} from "@/types/api";
import type {
  AutoencoderPredictRequest,
  AutoencoderPredictResponse,
  BenchmarkModel,
  CharacteristicRequest,
  CharacteristicResponse,
  FitRequest,
  FitResponse,
  HerdProfile,
  HerdProfileCreate,
  HerdProfileUploadResponse,
  MilkBotRunOptions,
  PresetDatasetKey,
  PresetDatasetResponse,
  PresetHerdStatsResponse,
  PresetPeriodKey,
  PresetSizeKey,
  PredictRequest,
  PredictResponse,
  TestIntervalRequest,
  TestIntervalResponse,
  YieldEstimateRequest,
  YieldEstimateResponse,
  ChallengeCreatePreset,
  ChallengeRead,
  SubmissionRead,
} from "@/types/api";

/* ------------------------------------------------------------------ */
/*  Generic fetch helpers                                              */
/* ------------------------------------------------------------------ */

async function apiFetch<T>(
  path: string,
  schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  body: unknown
): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`API error ${response.status} on ${path}: ${JSON.stringify(error)}`);
  }

  const data: unknown = await response.json();
  return schema.parse(data);
}

async function apiGet<T>(path: string, schema: z.ZodType<T, z.ZodTypeDef, unknown>): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`API error ${response.status} on ${path}: ${JSON.stringify(error)}`);
  }
  const data: unknown = await response.json();
  return schema.parse(data);
}

async function apiPut<T>(
  path: string,
  schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  body: unknown
): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`API error ${response.status} on ${path}: ${JSON.stringify(error)}`);
  }
  const data: unknown = await response.json();
  return schema.parse(data);
}

async function apiDelete(path: string): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, { method: "DELETE" });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`API error ${response.status} on ${path}: ${JSON.stringify(error)}`);
  }
}

/* ------------------------------------------------------------------ */
/*  Endpoint functions                                                 */
/* ------------------------------------------------------------------ */

export async function fitModel(request: FitRequest): Promise<FitResponse> {
  return apiFetch("/curves/fit", FitResponseSchema, request);
}

export async function getCharacteristic(
  request: CharacteristicRequest
): Promise<CharacteristicResponse> {
  return apiFetch("/curves/characteristic", CharacteristicResponseSchema, request);
}

export async function predictMilkbot(request: PredictRequest): Promise<PredictResponse> {
  return apiFetch("/curves/predict", PredictResponseSchema, request);
}

export async function getTestInterval(request: TestIntervalRequest): Promise<TestIntervalResponse> {
  return apiFetch("/curves/test-interval", TestIntervalResponseSchema, request);
}

export async function getIslc(request: YieldEstimateRequest): Promise<YieldEstimateResponse> {
  return apiFetch("/curves/islc", YieldEstimateResponseSchema, request);
}

export async function getBestPredict(
  request: YieldEstimateRequest
): Promise<YieldEstimateResponse> {
  return apiFetch("/curves/best-predict", YieldEstimateResponseSchema, request);
}

export async function predictAutoencoder(
  request: AutoencoderPredictRequest
): Promise<AutoencoderPredictResponse> {
  return apiFetch("/autoencoder/predict", AutoencoderPredictResponseSchema, request);
}

export async function healthCheck(): Promise<boolean> {
  const response = await fetch(`${getApiBaseUrl()}/health`);
  return response.ok;
}

/* ------------------------------------------------------------------ */
/*  Herd Profiles                                                      */
/* ------------------------------------------------------------------ */

export async function listHerdProfiles(): Promise<HerdProfile[]> {
  return apiGet("/herd-profiles/", HerdProfileListSchema);
}

export async function getHerdProfile(id: number): Promise<HerdProfile> {
  return apiGet(`/herd-profiles/${id}`, HerdProfileSchema);
}

export async function createHerdProfile(data: HerdProfileCreate): Promise<HerdProfile> {
  return apiFetch("/herd-profiles/", HerdProfileSchema, data);
}

export async function updateHerdProfile(id: number, data: HerdProfileCreate): Promise<HerdProfile> {
  return apiPut(`/herd-profiles/${id}`, HerdProfileSchema, data);
}

export async function deleteHerdProfile(id: number): Promise<void> {
  return apiDelete(`/herd-profiles/${id}`);
}

export async function getPresetDataset(
  dataset: PresetDatasetKey,
  size: PresetSizeKey,
  period: PresetPeriodKey
): Promise<PresetDatasetResponse> {
  return apiGet(`/datasets/presets/${dataset}/${size}/${period}`, PresetDatasetResponseSchema);
}

export async function getPresetHerdStats(
  dataset: PresetDatasetKey,
  size: PresetSizeKey,
  period: PresetPeriodKey,
  parity?: number
): Promise<PresetHerdStatsResponse> {
  const query = parity !== undefined ? `?parity=${parity}` : "";
  return apiGet(
    `/datasets/presets/${dataset}/${size}/${period}/herd-stats${query}`,
    PresetHerdStatsResponseSchema
  );
}

export async function uploadHerdProfileCsv(file: File): Promise<HerdProfileUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${getApiBaseUrl()}/herd-profiles/csv-preview`, {
    method: "POST",
    body: formData,
    // No Content-Type header - browser sets multipart boundary automatically
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`Upload error ${response.status}: ${JSON.stringify(error)}`);
  }
  const data: unknown = await response.json();
  return HerdProfileUploadResponseSchema.parse(data);
}

/* ------------------------------------------------------------------ */
/*  Benchmark - Challenges                                             */
/* ------------------------------------------------------------------ */

export async function createChallengePreset(data: ChallengeCreatePreset): Promise<ChallengeRead> {
  return apiFetch("/benchmark/challenges", ChallengeReadSchema, data);
}

export async function createChallengeUpload(
  name: string,
  testDayCsv: File,
  actualYieldsCsv: File
): Promise<ChallengeRead> {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("test_day_csv", testDayCsv);
  formData.append("actual_yields_csv", actualYieldsCsv);
  const response = await fetch(`${getApiBaseUrl()}/benchmark/challenges/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`Upload error ${response.status}: ${JSON.stringify(error)}`);
  }
  return ChallengeReadSchema.parse(await response.json());
}

export async function listChallenges(): Promise<ChallengeRead[]> {
  return apiGet("/benchmark/challenges", ChallengeListSchema);
}

export async function getChallenge(id: number): Promise<ChallengeRead> {
  return apiGet(`/benchmark/challenges/${id}`, ChallengeReadSchema);
}

export function exportChallengeUrl(id: number): string {
  return `${getApiBaseUrl()}/benchmark/challenges/${id}/export`;
}

/* ------------------------------------------------------------------ */
/*  Benchmark - Submissions                                            */
/* ------------------------------------------------------------------ */

export async function submitBoviModel(
  challengeId: number,
  data: {
    challenger: BenchmarkModel;
    benchmark: BenchmarkModel;
    challenger_options?: MilkBotRunOptions;
    benchmark_options?: MilkBotRunOptions;
    organization?: string;
    country?: string;
    notes?: string;
  }
): Promise<SubmissionRead> {
  return apiFetch(`/benchmark/challenges/${challengeId}/submissions`, SubmissionReadSchema, {
    submission_type: "bovi_model",
    ...data,
  });
}

export async function submitOwnMethod(
  challengeId: number,
  file: File,
  meta: {
    benchmark: BenchmarkModel;
    benchmark_options?: MilkBotRunOptions;
    organization?: string;
    country?: string;
    calculation_method?: string;
    notes?: string;
  }
): Promise<SubmissionRead> {
  const formData = new FormData();
  formData.append("file", file);
  if (meta.benchmark_options) {
    formData.append("benchmark_fitting", meta.benchmark_options.fitting);
    formData.append("benchmark_breed", meta.benchmark_options.breed);
    formData.append("benchmark_continent", meta.benchmark_options.continent);
  }
  Object.entries(meta).forEach(([k, v]) => {
    if (v && k !== "benchmark_options") formData.append(k, String(v));
  });
  const response = await fetch(
    `${getApiBaseUrl()}/benchmark/challenges/${challengeId}/submissions/upload`,
    {
      method: "POST",
      body: formData,
    }
  );
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`Upload error ${response.status}: ${JSON.stringify(error)}`);
  }
  return SubmissionReadSchema.parse(await response.json());
}

export async function listSubmissions(): Promise<SubmissionRead[]> {
  return apiGet("/benchmark/submissions", SubmissionListSchema);
}

export async function getSubmission(id: number): Promise<SubmissionRead> {
  return apiGet(`/benchmark/submissions/${id}`, SubmissionReadSchema);
}

export function downloadReportUrl(id: number): string {
  return `${getApiBaseUrl()}/benchmark/submissions/${id}/report`;
}
