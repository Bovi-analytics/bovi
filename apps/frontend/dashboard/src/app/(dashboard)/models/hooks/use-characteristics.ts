import { useQueries } from "@tanstack/react-query";
import { getCharacteristic } from "@/lib/api-client";
import type { Model, Characteristic } from "@/types/api";

const ALL_CHARACTERISTICS: Characteristic[] = [
  "peak_yield",
  "time_to_peak",
  "cumulative_milk_yield",
  "persistency",
];

interface UseCharacteristicsParams {
  readonly model: Model;
  readonly dim: readonly number[];
  readonly milkrecordings: readonly number[];
  readonly parity: number;
}

export function useCharacteristics({
  model,
  dim,
  milkrecordings,
  parity,
}: UseCharacteristicsParams) {
  const results = useQueries({
    queries: ALL_CHARACTERISTICS.map((characteristic) => ({
      queryKey: ["characteristic", model, characteristic, dim, milkrecordings] as const,
      queryFn: () =>
        getCharacteristic({
          model,
          characteristic,
          dim: [...dim],
          milkrecordings: [...milkrecordings],
          parity,
        }),
    })),
  });

  const characteristics = ALL_CHARACTERISTICS.map((name, i) => ({
    name,
    value: results[i].data?.value ?? null,
    isLoading: results[i].isLoading,
    error: results[i].error,
  }));

  return {
    characteristics,
    isLoading: results.some((r) => r.isLoading),
  };
}
