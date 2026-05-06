import { useQueries } from "@tanstack/react-query";
import { fitModel } from "@/lib/api-client";
import type { Model } from "@/types/api";

interface UseComparisonParams {
  readonly models: readonly Model[];
  readonly dim: readonly number[];
  readonly milkrecordings: readonly number[];
}

export function useComparison({ models, dim, milkrecordings }: UseComparisonParams) {
  return useQueries({
    queries: models.map((model) => ({
      queryKey: ["fit", model, dim, milkrecordings] as const,
      queryFn: () =>
        fitModel({
          model,
          dim: [...dim],
          milkrecordings: [...milkrecordings],
        }),
    })),
  });
}
