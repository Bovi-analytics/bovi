import { useQuery } from "@tanstack/react-query";
import { getCharacteristicsBatch } from "@/lib/api-client";
import type { Model, Characteristic, MilkBotRunOptions } from "@/types/api";

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
  readonly milkbotOptions: MilkBotRunOptions;
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
  milkbotOptions,
}: UseAllCharacteristicsParams): ModelCharacteristics[] {
  const batchResult = useQuery({
    queryKey: ["characteristics", models, dim, milkrecordings, parity, milkbotOptions] as const,
    enabled: models.length > 0,
    queryFn: () =>
      getCharacteristicsBatch({
        items: models.flatMap((model) =>
          ALL_CHARACTERISTICS.map((characteristic) => ({
            id: `${model}:${characteristic}`,
            model,
            characteristic,
            dim: [...dim],
            milkrecordings: [...milkrecordings],
            parity,
            ...(model === "milkbot" ? milkbotOptions : {}),
          }))
        ),
      }),
  });

  const valuesById = new Map(
    batchResult.data?.results.map((item) => [item.id, item.value] as const) ?? []
  );
  return models.map((model, modelIndex) => {
    const base = modelIndex * ALL_CHARACTERISTICS.length;
    const valueFor = (characteristic: Characteristic, offset: number) =>
      valuesById.get(`${model}:${characteristic}`) ??
      batchResult.data?.results[base + offset]?.value ??
      null;

    return {
      model,
      peakYield: valueFor("peak_yield", 0),
      timeToPeak: valueFor("time_to_peak", 1),
      cumulativeYield: valueFor("cumulative_milk_yield", 2),
      persistency: valueFor("persistency", 3),
      isLoading: batchResult.isLoading,
    };
  });
}
