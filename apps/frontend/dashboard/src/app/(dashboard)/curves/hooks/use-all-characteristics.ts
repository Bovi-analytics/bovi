import { useQueries } from "@tanstack/react-query";
import { getCharacteristic } from "@/lib/api-client";
import type { Model, Characteristic } from "@/types/api";

const ALL_CHARACTERISTICS: Characteristic[] = [
  "peak_yield",
  "time_to_peak",
  "cumulative_milk_yield",
  "persistency",
];

interface UseAllCharacteristicsParams {
  readonly models: readonly Model[];
  readonly dim: readonly number[];
  readonly milkrecordings: readonly number[];
  readonly parity: number;
}

interface ModelCharacteristics {
  readonly model: Model;
  readonly peakYield: number | null;
  readonly timeToPeak: number | null;
  readonly cumulativeYield: number | null;
  readonly persistency: number | null;
  readonly isLoading: boolean;
}

/**
 * Fetch characteristics for multiple models in a single hook call.
 * This avoids calling hooks inside loops (React rules of hooks).
 */
export function useAllCharacteristics({
  models,
  dim,
  milkrecordings,
  parity,
}: UseAllCharacteristicsParams): ModelCharacteristics[] {
  // Create one query per (model, characteristic) pair
  const queries = models.flatMap((model) =>
    ALL_CHARACTERISTICS.map((characteristic) => ({
      queryKey: ["characteristic", model, characteristic, dim, milkrecordings] as const,
      queryFn: () =>
        getCharacteristic({
          model,
          characteristic,
          dim: [...dim],
          milkrecordings: [...milkrecordings],
          parity,
        }),
    }))
  );

  const results = useQueries({ queries });

  // Group results back by model (4 characteristics per model)
  return models.map((model, modelIndex) => {
    const base = modelIndex * ALL_CHARACTERISTICS.length;
    return {
      model,
      peakYield: results[base]?.data?.value ?? null,
      timeToPeak: results[base + 1]?.data?.value ?? null,
      cumulativeYield: results[base + 2]?.data?.value ?? null,
      persistency: results[base + 3]?.data?.value ?? null,
      isLoading: ALL_CHARACTERISTICS.some((_, i) => results[base + i]?.isLoading),
    };
  });
}
