import type { z } from "zod";
import { getApiBaseUrl } from "@/lib/env";
import {
  FitResponseSchema,
  CharacteristicResponseSchema,
  PredictResponseSchema,
  TestIntervalResponseSchema,
} from "@/types/api";
import type {
  FitRequest,
  FitResponse,
  CharacteristicRequest,
  CharacteristicResponse,
  PredictRequest,
  PredictResponse,
  TestIntervalRequest,
  TestIntervalResponse,
} from "@/types/api";

/* ------------------------------------------------------------------ */
/*  Generic fetch helper — all API calls go through this              */
/* ------------------------------------------------------------------ */

async function apiFetch<T>(path: string, schema: z.ZodType<T>, body: unknown): Promise<T> {
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

/* ------------------------------------------------------------------ */
/*  Endpoint functions                                                 */
/* ------------------------------------------------------------------ */

export async function fitModel(request: FitRequest): Promise<FitResponse> {
  return apiFetch("/fit", FitResponseSchema, request);
}

export async function getCharacteristic(
  request: CharacteristicRequest
): Promise<CharacteristicResponse> {
  return apiFetch("/characteristic", CharacteristicResponseSchema, request);
}

export async function predictMilkbot(request: PredictRequest): Promise<PredictResponse> {
  return apiFetch("/predict", PredictResponseSchema, request);
}

export async function getTestInterval(request: TestIntervalRequest): Promise<TestIntervalResponse> {
  return apiFetch("/test-interval", TestIntervalResponseSchema, request);
}

export async function healthCheck(): Promise<boolean> {
  const response = await fetch(`${getApiBaseUrl()}/`);
  return response.ok;
}
