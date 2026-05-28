import type { z } from "zod";
import { getBackendAccessToken, handleUnauthorizedResponse } from "@/lib/auth/service";
import { getApiBaseUrl } from "@/lib/env";
import {
  AutoencoderPredictResponseSchema,
  CharacteristicBatchResponseSchema,
  CharacteristicResponseSchema,
  FitResponseSchema,
  HerdProfileListSchema,
  HerdProfileSchema,
  HerdProfileUploadResponseSchema,
  PresetCountsResponseSchema,
  PresetDatasetResponseSchema,
  PresetHerdStatsResponseSchema,
  PredictResponseSchema,
  TestIntervalResponseSchema,
  YieldEstimateResponseSchema,
  ChallengeListSchema,
  ChallengeDetailSchema,
  ChallengeReadSchema,
  SubmissionListSchema,
  SubmissionReadSchema,
  AdminOverviewResponseSchema,
} from "@/types/api";
import type {
  AdminDataCategory,
  AdminOverviewResponse,
  AutoencoderPredictRequest,
  AutoencoderPredictResponse,
  BenchmarkModel,
  CharacteristicRequest,
  CharacteristicBatchRequest,
  CharacteristicBatchResponse,
  CharacteristicResponse,
  FitRequest,
  FitResponse,
  HerdProfile,
  HerdProfileCreate,
  HerdProfileUploadResponse,
  MilkBotRunOptions,
  PresetCountsResponse,
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
  ChallengeDatasetSource,
  ChallengeDetail,
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

async function apiGet<T>(path: string, schema: z.ZodType<T, z.ZodTypeDef, unknown>): Promise<T> {
  const headers = await jsonHeaders();
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "GET",
    headers,
  });
  await ensureOk(response, path);
  const data: unknown = await response.json();
  return schema.parse(data);
}

async function apiPut<T>(
  path: string,
  schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  body: unknown
): Promise<T> {
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

async function apiPatch<T>(path: string, body: unknown): Promise<T> {
  const headers = await jsonHeaders();
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "PATCH",
    headers,
    body: JSON.stringify(body),
  });
  await ensureOk(response, path);
  return (await response.json()) as T;
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

export interface OrganizationRead {
  id: number;
  name: string;
  role: string | null;
  created_by_user_id?: number | null;
  source_entra_tenant_id?: string | null;
  source_domain?: string | null;
  source_display_name?: string | null;
}

export interface OrganizationMemberRead {
  user_id: number;
  email: string | null;
  name: string | null;
  role: string;
}

export interface OrganizationInviteRead {
  id: number;
  organization_id: number;
  created_by_user_id: number | null;
  created_at: string | null;
  expires_at: string;
  revoked_at: string | null;
  accepted_count: number;
  last_accepted_at: string | null;
}

export interface OrganizationInviteCreateResponse extends OrganizationInviteRead {
  token: string;
}

export interface OrganizationListOptions {
  scope?: "organization" | "mine";
  sort?: "created_at" | "name" | "user";
  direction?: "asc" | "desc";
  q?: string;
}

export interface AdminOverviewOptions {
  organizationId?: number | "all";
  category?: AdminDataCategory | "all";
  userId?: number;
  q?: string;
  from?: string;
  to?: string;
  sort?: "created_at" | "organization" | "user" | "category" | "status";
  direction?: "asc" | "desc";
  limit?: number;
}

export async function listOrganizations(): Promise<OrganizationRead[]> {
  const response = await fetch(`${getApiBaseUrl()}/organizations`, {
    method: "GET",
    headers: await jsonHeaders(),
  });
  await ensureOk(response, "/organizations");
  return (await response.json()) as OrganizationRead[];
}

export async function createOrganization(name: string): Promise<OrganizationRead> {
  const response = await fetch(`${getApiBaseUrl()}/organizations`, {
    method: "POST",
    headers: await jsonHeaders(),
    body: JSON.stringify({ name }),
  });
  await ensureOk(response, "/organizations");
  return (await response.json()) as OrganizationRead;
}

export async function updateOrganization(id: number, name: string): Promise<OrganizationRead> {
  return apiPatch<OrganizationRead>(`/organizations/${id}`, { name });
}

export async function listOrganizationMembers(id: number): Promise<OrganizationMemberRead[]> {
  const response = await fetch(`${getApiBaseUrl()}/organizations/${id}/members`, {
    method: "GET",
    headers: await jsonHeaders(),
  });
  await ensureOk(response, `/organizations/${id}/members`);
  return (await response.json()) as OrganizationMemberRead[];
}

export async function removeOrganizationMember(id: number, userId: number): Promise<void> {
  return apiDelete(`/organizations/${id}/members/${userId}`);
}

export async function listOrganizationInvites(id: number): Promise<OrganizationInviteRead[]> {
  const response = await fetch(`${getApiBaseUrl()}/organizations/${id}/invites`, {
    method: "GET",
    headers: await jsonHeaders(),
  });
  await ensureOk(response, `/organizations/${id}/invites`);
  return (await response.json()) as OrganizationInviteRead[];
}

export async function createOrganizationInvite(
  id: number
): Promise<OrganizationInviteCreateResponse> {
  const response = await fetch(`${getApiBaseUrl()}/organizations/${id}/invites`, {
    method: "POST",
    headers: await jsonHeaders(),
  });
  await ensureOk(response, `/organizations/${id}/invites`);
  return (await response.json()) as OrganizationInviteCreateResponse;
}

export async function revokeOrganizationInvite(id: number, inviteId: number): Promise<void> {
  return apiDelete(`/organizations/${id}/invites/${inviteId}`);
}

export async function acceptInvite(token: string): Promise<OrganizationRead> {
  const response = await fetch(`${getApiBaseUrl()}/invites/${encodeURIComponent(token)}/accept`, {
    method: "POST",
    headers: await jsonHeaders(),
  });
  await ensureOk(response, "/invites/accept");
  return (await response.json()) as OrganizationRead;
}

export async function fitModel(request: FitRequest): Promise<FitResponse> {
  return apiFetch("/curves/fit", FitResponseSchema, request);
}

export async function getCharacteristic(
  request: CharacteristicRequest
): Promise<CharacteristicResponse> {
  return apiFetch("/curves/characteristic", CharacteristicResponseSchema, request);
}

export async function getCharacteristicsBatch(
  request: CharacteristicBatchRequest
): Promise<CharacteristicBatchResponse> {
  return apiFetch("/curves/characteristic/batch", CharacteristicBatchResponseSchema, request);
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

function organizationQuery(
  organizationId: number | "all",
  options: OrganizationListOptions = {}
): string {
  const params = new URLSearchParams({ organization_id: String(organizationId) });
  Object.entries(options).forEach(([key, value]) => {
    if (value) {
      params.set(key, value);
    }
  });
  return `?${params.toString()}`;
}

function listQueryKey(options: OrganizationListOptions): string {
  const params = new URLSearchParams();
  Object.entries(options).forEach(([key, value]) => {
    if (value) {
      params.set(key, value);
    }
  });
  return params.toString();
}

export function listOptionsKey(options: OrganizationListOptions = {}): string {
  return listQueryKey(options);
}

function adminOverviewQuery(options: AdminOverviewOptions = {}): string {
  const params = new URLSearchParams();
  if (options.organizationId !== undefined) {
    params.set("organization_id", String(options.organizationId));
  }
  if (options.category && options.category !== "all") {
    params.set("category", options.category);
  }
  if (options.userId !== undefined) {
    params.set("user_id", String(options.userId));
  }
  if (options.q) {
    params.set("q", options.q);
  }
  if (options.from) {
    params.set("from", options.from);
  }
  if (options.to) {
    params.set("to", options.to);
  }
  if (options.sort) {
    params.set("sort", options.sort);
  }
  if (options.direction) {
    params.set("direction", options.direction);
  }
  if (options.limit !== undefined) {
    params.set("limit", String(options.limit));
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}

export function adminOverviewOptionsKey(options: AdminOverviewOptions = {}): string {
  return adminOverviewQuery(options);
}

/* ------------------------------------------------------------------ */
/*  Admin                                                             */
/* ------------------------------------------------------------------ */

export async function listAdminSubmissionsOverview(
  options: AdminOverviewOptions = {}
): Promise<AdminOverviewResponse> {
  return apiGet(
    `/admin/submissions-overview${adminOverviewQuery(options)}`,
    AdminOverviewResponseSchema
  );
}

/* ------------------------------------------------------------------ */
/*  Herd Profiles                                                      */
/* ------------------------------------------------------------------ */

export async function listHerdProfiles(
  organizationId: number | "all",
  options: OrganizationListOptions = {}
): Promise<HerdProfile[]> {
  return apiGet(
    `/herd-profiles/${organizationQuery(organizationId, options)}`,
    HerdProfileListSchema
  );
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

export async function getPresetCounts(dataset: PresetDatasetKey): Promise<PresetCountsResponse> {
  return apiGet(`/datasets/presets/${dataset}/counts`, PresetCountsResponseSchema);
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

export async function uploadHerdProfileCsv(
  file: File,
  organizationId: number,
  columnMapping?: Record<string, string>
): Promise<HerdProfileUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("organization_id", String(organizationId));
  if (columnMapping) {
    formData.append("column_mapping", JSON.stringify(columnMapping));
  }
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
  actualYieldsCsv: File,
  organizationId: number
): Promise<ChallengeRead> {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("organization_id", String(organizationId));
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

export async function listChallenges(
  organizationId: number | "all",
  options: OrganizationListOptions = {}
): Promise<ChallengeRead[]> {
  return apiGet(
    `/benchmark/challenges${organizationQuery(organizationId, options)}`,
    ChallengeListSchema
  );
}

export async function createChallengeFromSavedDataset(
  name: string,
  cowMetadata: ChallengeDetail["cow_metadata"],
  actualYields: NonNullable<ChallengeDetail["actual_yields"]>,
  organizationId: number,
  datasetSources?: ChallengeDatasetSource[]
): Promise<ChallengeRead> {
  return apiFetch("/benchmark/challenges/saved-dataset", ChallengeReadSchema, {
    name,
    cow_metadata: cowMetadata,
    actual_yields: actualYields,
    organization_id: organizationId,
    dataset_sources: datasetSources,
  });
}

export async function getChallenge(id: number): Promise<ChallengeDetail> {
  return apiGet(`/benchmark/challenges/${id}`, ChallengeDetailSchema);
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

export async function listSubmissions(
  organizationId: number | "all",
  options: OrganizationListOptions = {}
): Promise<SubmissionRead[]> {
  return apiGet(
    `/benchmark/submissions${organizationQuery(organizationId, options)}`,
    SubmissionListSchema
  );
}

export async function getSubmission(id: number): Promise<SubmissionRead> {
  return apiGet(`/benchmark/submissions/${id}`, SubmissionReadSchema);
}

export async function downloadSubmissionReport(id: number): Promise<void> {
  await downloadBlob(`/benchmark/submissions/${id}/report`, `benchmark_report_${id}.pdf`);
}
