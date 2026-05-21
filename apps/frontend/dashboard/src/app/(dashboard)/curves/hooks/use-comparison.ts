import { useQueries } from "@tanstack/react-query";
import { fitModel } from "@/lib/api-client";
import type { MilkBotRunOptions, Model } from "@/types/api";

interface UseComparisonParams {
  readonly models: readonly Model[];
  readonly dim: readonly number[];
  readonly milkrecordings: readonly number[];
  readonly parity: number;
  readonly milkbotOptions: MilkBotRunOptions;
}

export function useComparison({
  models,
  dim,
  milkrecordings,
  parity,
  milkbotOptions,
}: UseComparisonParams) {
  return useQueries({
    queries: models.map((model) => ({
      queryKey: [
        "fit",
        model,
        dim,
        milkrecordings,
        model === "milkbot" ? milkbotOptions : null,
      ] as const,
      queryFn: () =>
        fitModel({
          model,
          dim: [...dim],
          milkrecordings: [...milkrecordings],
          ...(model === "milkbot" ? { parity } : {}),
          ...(model === "milkbot" ? milkbotOptions : {}),
        }),
    })),
  });
}
