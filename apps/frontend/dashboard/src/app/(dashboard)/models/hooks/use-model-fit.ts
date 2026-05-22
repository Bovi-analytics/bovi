import { useQuery } from "@tanstack/react-query";
import { fitModel } from "@/lib/api-client";
import type { Model, FitResponse, MilkBotRunOptions } from "@/types/api";

interface UseModelFitParams {
  readonly model: Model;
  readonly dim: readonly number[];
  readonly milkrecordings: readonly number[];
  readonly milkbotOptions?: MilkBotRunOptions;
}

export function useModelFit({ model, dim, milkrecordings, milkbotOptions }: UseModelFitParams) {
  return useQuery<FitResponse>({
    queryKey: ["fit", model, dim, milkrecordings, model === "milkbot" ? milkbotOptions : null],
    queryFn: () =>
      fitModel({
        model,
        dim: [...dim],
        milkrecordings: [...milkrecordings],
        ...(model === "milkbot" && milkbotOptions ? milkbotOptions : {}),
      }),
  });
}
