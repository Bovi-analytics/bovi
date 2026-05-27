import { useQuery } from "@tanstack/react-query";
import { predictAutoencoder } from "@/lib/api-client";
import {
  buildAutoencoderPredictRequest,
  type AutoencoderPredictRequestInput,
} from "@/lib/autoencoder-request";
import type { AutoencoderPredictResponse } from "@/types/api";

interface UseAutoencoderPredictParams extends AutoencoderPredictRequestInput {
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
      const request = buildAutoencoderPredictRequest({
        milk,
        parity,
        herdId,
        events,
        herdStats,
        imputationMethod,
      });
      return predictAutoencoder(request);
    },
    enabled,
  });
}
