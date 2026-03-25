import { useQuery } from "@tanstack/react-query";
import { fitModel } from "@/lib/api-client";
import type { Model, FitResponse } from "@/types/api";

interface UseModelFitParams {
  readonly model: Model;
  readonly dim: readonly number[];
  readonly milkrecordings: readonly number[];
}

export function useModelFit({ model, dim, milkrecordings }: UseModelFitParams) {
  return useQuery<FitResponse>({
    queryKey: ["fit", model, dim, milkrecordings],
    queryFn: () =>
      fitModel({
        model,
        dim: [...dim],
        milkrecordings: [...milkrecordings],
      }),
  });
}
