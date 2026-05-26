import { useQuery } from "@tanstack/react-query";
import { getCharacteristicsBatch } from "@/lib/api-client";
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
  const result = useQuery({
    queryKey: ["characteristics", model, dim, milkrecordings, parity] as const,
    queryFn: () =>
      getCharacteristicsBatch({
        items: ALL_CHARACTERISTICS.map((characteristic) => ({
          id: characteristic,
          model,
          characteristic,
          dim: [...dim],
          milkrecordings: [...milkrecordings],
          parity,
        })),
      }),
  });

  const valuesByName = new Map(
    result.data?.results.map((item) => [item.id, item.value] as const) ?? []
  );
  const characteristics = ALL_CHARACTERISTICS.map((name, i) => ({
    name,
    value: valuesByName.get(name) ?? result.data?.results[i]?.value ?? null,
    isLoading: result.isLoading,
    error: result.error,
  }));

  return {
    characteristics,
    isLoading: result.isLoading,
  };
}
