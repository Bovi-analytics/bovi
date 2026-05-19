import type { z } from "zod";
import { getBackendAccessToken, handleUnauthorizedResponse } from "@/lib/auth/service";
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

async function apiFetch<T>(path: string, schema: z.ZodType<T>, body: unknown): Promise<T> {
  const headers = await jsonHeaders();
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  await ensureOk(response, path);

  const data: unknown = await response.json();
  return schema.parse(data);
}

async function apiGet<T>(path: string, schema: z.ZodType<T>): Promise<T> {
  const headers = await jsonHeaders();
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "GET",
    headers,
  });
  await ensureOk(response, path);
  const data: unknown = await response.json();
  return schema.parse(data);
}

async function apiPut<T>(path: string, schema: z.ZodType<T>, body: unknown): Promise<T> {
  const headers = await jsonHeaders();
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "PUT",
    headers,
    body: JSON.stringify(body),
  });
  await ensureOk(response, path);
  const data: unknown = await response.json();
  return schema.parse(data);
}

async function apiDelete(path: string): Promise<void> {
  const headers = await authHeaders();
  const response = await fetch(`${getApiBaseUrl()}${path}`, { method: "DELETE", headers });
  await ensureOk(response, path);
}

async function authHeaders(): Promise<HeadersInit> {
  const token = await getBackendAccessToken();
  return { Authorization: `Bearer ${token}` };
}

async function jsonHeaders(): Promise<HeadersInit> {
  return { "Content-Type": "application/json", ...(await authHeaders()) };
}

async function ensureOk(response: Response, path: string): Promise<void> {
  if (response.ok) return;
  if (response.status === 401) {
    handleUnauthorizedResponse();
  }
  const error = await response.json().catch(() => ({}));
  throw new Error(`API error ${response.status} on ${path}: ${JSON.stringify(error)}`);
}

function filenameFromContentDisposition(header: string | null, fallback: string): string {
  if (!header) return fallback;
  const match = /filename="?([^";]+)"?/i.exec(header);
  return match?.[1] ?? fallback;
}

async function downloadBlob(path: string, fallbackFilename: string): Promise<void> {
  const headers = await authHeaders();
  const response = await fetch(`${getApiBaseUrl()}${path}`, { headers });
  await ensureOk(response, path);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filenameFromContentDisposition(
    response.headers.get("Content-Disposition"),
    fallbackFilename
  );
  anchor.click();
  URL.revokeObjectURL(url);
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
  const response = await fetch(`${getApiBaseUrl()}/`);
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
  const headers = await authHeaders();
  const response = await fetch(`${getApiBaseUrl()}/herd-profiles/csv-preview`, {
    method: "POST",
    headers,
    body: formData,
    // No Content-Type header - browser sets multipart boundary automatically
  });
  await ensureOk(response, "/herd-profiles/csv-preview");
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
  const headers = await authHeaders();
  const response = await fetch(`${getApiBaseUrl()}/benchmark/challenges/upload`, {
    method: "POST",
    headers,
    body: formData,
  });
  await ensureOk(response, "/benchmark/challenges/upload");
  return ChallengeReadSchema.parse(await response.json());
}

export async function listChallenges(): Promise<ChallengeRead[]> {
  return apiGet("/benchmark/challenges", ChallengeListSchema);
}

export async function getChallenge(id: number): Promise<ChallengeRead> {
  return apiGet(`/benchmark/challenges/${id}`, ChallengeReadSchema);
}

export async function downloadChallengeExport(id: number): Promise<void> {
  await downloadBlob(`/benchmark/challenges/${id}/export`, `challenge_${id}.csv`);
}

/* ------------------------------------------------------------------ */
/*  Benchmark - Submissions                                            */
/* ------------------------------------------------------------------ */

export async function submitBoviModel(
  challengeId: number,
  data: {
    challenger: BenchmarkModel;
    benchmark: BenchmarkModel;
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
    organization?: string;
    country?: string;
    calculation_method?: string;
    notes?: string;
  }
): Promise<SubmissionRead> {
  const formData = new FormData();
  formData.append("file", file);
  Object.entries(meta).forEach(([k, v]) => {
    if (v) formData.append(k, String(v));
  });
  const headers = await authHeaders();
  const response = await fetch(
    `${getApiBaseUrl()}/benchmark/challenges/${challengeId}/submissions/upload`,
    {
      method: "POST",
      headers,
      body: formData,
    }
  );
  await ensureOk(response, `/benchmark/challenges/${challengeId}/submissions/upload`);
  return SubmissionReadSchema.parse(await response.json());
}

export async function listSubmissions(): Promise<SubmissionRead[]> {
  return apiGet("/benchmark/submissions", SubmissionListSchema);
}

export async function getSubmission(id: number): Promise<SubmissionRead> {
  return apiGet(`/benchmark/submissions/${id}`, SubmissionReadSchema);
}

export async function downloadSubmissionReport(id: number): Promise<void> {
  await downloadBlob(`/benchmark/submissions/${id}/report`, `benchmark_report_${id}.pdf`);
}
