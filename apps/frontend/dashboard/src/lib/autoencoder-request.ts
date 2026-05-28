import type { AutoencoderPredictRequest, ImputationMethod } from "@/types/api";

export interface AutoencoderPredictRequestInput {
  readonly milk?: readonly (number | null)[];
  readonly dim?: readonly number[];
  readonly milkrecordings?: readonly number[];
  readonly parity: number;
  readonly herdId?: number;
  readonly events?: readonly string[];
  readonly herdStats?: readonly number[];
  readonly imputationMethod?: ImputationMethod;
}

export function buildAutoencoderPredictRequest({
  milk,
  dim,
  milkrecordings,
  parity,
  herdId,
  events,
  herdStats,
  imputationMethod,
}: AutoencoderPredictRequestInput): AutoencoderPredictRequest {
  return {
    ...(milk !== undefined && { milk: [...milk] }),
    ...(dim !== undefined && { dim: [...dim] }),
    ...(milkrecordings !== undefined && { milkrecordings: [...milkrecordings] }),
    parity,
    ...(herdId !== undefined && { herd_id: herdId }),
    ...(events !== undefined && { events: [...events] }),
    ...(herdStats !== undefined && { herd_stats: [...herdStats] }),
    ...(imputationMethod !== undefined && { imputation_method: imputationMethod }),
  };
}
