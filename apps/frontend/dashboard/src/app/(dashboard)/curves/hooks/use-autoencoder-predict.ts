import { useQuery } from "@tanstack/react-query";
import { predictAutoencoder } from "@/lib/api-client";
import type {
  AutoencoderPredictRequest,
  AutoencoderPredictResponse,
  ImputationMethod,
} from "@/types/api";

interface UseAutoencoderPredictParams {
  readonly milk: readonly (number | null)[];
  readonly parity: number;
  readonly events?: readonly string[];
  readonly herdStats?: readonly number[];
  readonly imputationMethod?: ImputationMethod;
  readonly enabled?: boolean;
}

export function useAutoencoderPredict({
  milk,
  parity,
  events,
  herdStats,
  imputationMethod,
  enabled = true,
}: UseAutoencoderPredictParams) {
  return useQuery<AutoencoderPredictResponse>({
    queryKey: ["autoencoder-predict", milk, parity, events, herdStats, imputationMethod],
    queryFn: () => {
      const request: AutoencoderPredictRequest = {
        milk: [...milk],
        parity,
        ...(events !== undefined && { events: [...events] }),
        ...(herdStats !== undefined && { herd_stats: [...herdStats] }),
        ...(imputationMethod !== undefined && { imputation_method: imputationMethod }),
      };
      return predictAutoencoder(request);
    },
    enabled,
  });
}
