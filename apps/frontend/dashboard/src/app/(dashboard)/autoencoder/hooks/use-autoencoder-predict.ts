import { useQuery } from "@tanstack/react-query";
import { predictAutoencoder } from "@/lib/api-client";
import type { AutoencoderPredictRequest, AutoencoderPredictResponse } from "@/types/api";

interface UseAutoencoderPredictParams {
  readonly milk: readonly (number | null)[];
  readonly parity: number;
  readonly herdId?: number;
  readonly events?: readonly string[];
  readonly herdStats?: readonly number[];
  readonly imputationMethod?: string;
  readonly enabled?: boolean;
}

export function useAutoencoderPredict({
  milk,
  parity,
  herdId,
  events,
  herdStats,
  imputationMethod,
  enabled = true,
}: UseAutoencoderPredictParams) {
  return useQuery<AutoencoderPredictResponse>({
    queryKey: ["autoencoder-predict", milk, parity, herdId, events, herdStats, imputationMethod],
    queryFn: () => {
      const request: AutoencoderPredictRequest = {
        milk: [...milk],
        parity,
        ...(herdId !== undefined && { herd_id: herdId }),
        ...(events !== undefined && { events: [...events] }),
        ...(herdStats !== undefined && { herd_stats: [...herdStats] }),
        ...(imputationMethod !== undefined && { imputation_method: imputationMethod as AutoencoderPredictRequest["imputation_method"] }),
      };
      return predictAutoencoder(request);
    },
    enabled,
  });
}
