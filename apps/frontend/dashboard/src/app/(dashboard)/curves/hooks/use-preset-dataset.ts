import { useQuery } from "@tanstack/react-query";
import { getPresetDataset } from "@/lib/api-client";
import type {
  PresetDatasetKey,
  PresetDatasetResponse,
  PresetPeriodKey,
  PresetSizeKey,
} from "@/types/api";

export function usePresetDataset(
  dataset: PresetDatasetKey | null,
  size: PresetSizeKey,
  period: PresetPeriodKey
): {
  data: PresetDatasetResponse | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
} {
  const query = useQuery({
    queryKey: ["preset-dataset", dataset, size, period] as const,
    queryFn: () => getPresetDataset(dataset!, size, period),
    enabled: dataset !== null,
    staleTime: Infinity,
    retry: 1,
  });
  return {
    data: query.data,
    isLoading: query.isLoading && query.fetchStatus !== "idle",
    isError: query.isError,
    error: query.error,
  };
}
