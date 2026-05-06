import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { getPresetHerdStats } from "@/lib/api-client";
import { HERD_STATS_METADATA } from "@/data/herd-stats-metadata";
import type {
  PresetDatasetKey,
  PresetHerdStatsResponse,
  PresetPeriodKey,
  PresetSizeKey,
} from "@/types/api";

interface UsePresetHerdStatsResult {
  readonly data: PresetHerdStatsResponse | undefined;
  readonly statsArray: number[] | undefined;
  readonly isLoading: boolean;
  readonly isError: boolean;
  readonly error: Error | null;
}

/**
 * Fetch the 10 normalized herd_stats computed from a preset dataset slice.
 *
 * Returns both the raw response and a `statsArray` ordered to match
 * HERD_STATS_METADATA — i.e. ready to send as the autoencoder `herd_stats`
 * field. Missing keys fall back to the metadata default so the array always
 * has length 10.
 */
export function usePresetHerdStats(
  dataset: PresetDatasetKey | null,
  size: PresetSizeKey,
  period: PresetPeriodKey,
  parity?: number
): UsePresetHerdStatsResult {
  const query = useQuery({
    queryKey: ["preset-herd-stats", dataset, size, period, parity ?? null] as const,
    queryFn: () => getPresetHerdStats(dataset!, size, period, parity),
    enabled: dataset !== null,
    staleTime: Infinity,
    retry: 1,
  });

  const statsArray = useMemo(() => {
    if (!query.data) return undefined;
    return HERD_STATS_METADATA.map(
      (m) => query.data!.stats[m.name] ?? m.default
    );
  }, [query.data]);

  return {
    data: query.data,
    statsArray,
    isLoading: query.isLoading && query.fetchStatus !== "idle",
    isError: query.isError,
    error: query.error,
  };
}
