import type { AutoencoderPredictRequest, ImputationMethod } from "@/types/api";

export interface AutoencoderPredictRequestInput {
  readonly milk: readonly (number | null)[];
  readonly parity: number;
  readonly herdId?: number;
  readonly events?: readonly string[];
  readonly herdStats?: readonly number[];
  readonly imputationMethod?: ImputationMethod;
}

export function buildAutoencoderPredictRequest({
  milk,
  parity,
  herdId,
  events,
  herdStats,
  imputationMethod,
}: AutoencoderPredictRequestInput): AutoencoderPredictRequest {
  return {
    milk: [...milk],
    parity,
    ...(herdId !== undefined && { herd_id: herdId }),
    ...(events !== undefined && { events: [...events] }),
    ...(herdStats !== undefined && { herd_stats: [...herdStats] }),
    ...(imputationMethod !== undefined && { imputation_method: imputationMethod }),
  };
}
