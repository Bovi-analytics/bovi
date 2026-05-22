import { useQuery } from "@tanstack/react-query";
import { getPresetCounts } from "@/lib/api-client";
import type { PresetCountsResponse, PresetDatasetKey } from "@/types/api";

export function usePresetCounts(dataset: PresetDatasetKey | null): {
  data: PresetCountsResponse | undefined;
  isLoading: boolean;
  isError: boolean;
} {
  const query = useQuery({
    queryKey: ["preset-counts", dataset] as const,
    queryFn: () => getPresetCounts(dataset!),
    enabled: dataset !== null,
    staleTime: Infinity,
    retry: 1,
  });
  return {
    data: query.data,
    isLoading: query.isLoading && query.fetchStatus !== "idle",
    isError: query.isError,
  };
}
