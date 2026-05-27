import type { ChallengeDatasetSource, ChallengeDatasetStats, ChallengeRead } from "@/types/api";

export const BENCHMARK_DATASET_LABELS: Record<string, string> = {
  icar: "Demo dataset",
  upload: "Custom upload",
  saved_upload: "Saved dataset",
};

export const MODEL_LABELS: Record<string, string> = {
  tim: "TIM (Test Interval Method)",
  islc: "ISLC (Standard Lactation Curve interpolation)",
  best_predict: "Best Prediction",
  wood: "Wood",
  wilmink: "Wilmink",
  ali_schaeffer: "Ali-Schaeffer",
  fischer: "Fischer",
  milkbot: "MilkBot",
  autoencoder: "AI autoencoder",
};

export function getBenchmarkDatasetLabel(challenge: Pick<ChallengeRead, "dataset">): string {
  return BENCHMARK_DATASET_LABELS[challenge.dataset] ?? challenge.dataset;
}

export function formatCount(value: number | null | undefined, label: string): string | null {
  if (value === null || value === undefined) return null;
  return `${value.toLocaleString()} ${label}`;
}

export function formatDatasetStats(stats: ChallengeDatasetStats | null | undefined): string {
  const parts = [
    formatCount(stats?.lactation_count, "lactations"),
    formatCount(stats?.test_day_row_count, "test-day rows"),
    formatCount(stats?.actual_yield_count, "ALY rows"),
  ].filter((part): part is string => Boolean(part));
  return parts.length > 0 ? parts.join(" · ") : "Dataset contents unavailable";
}

export function formatDatasetSources(
  sources: readonly ChallengeDatasetSource[] | null | undefined
): string {
  if (!sources || sources.length === 0) return "Sources unavailable";
  return sources
    .map((source) => `${source.label}: ${source.filename ?? "Unknown source"}`)
    .join(" · ");
}

export function datasetStatsFromChallengeDetail(
  cowMetadata: Record<string, { dim: number[]; herd_id?: number | null }>,
  actualYields: Record<string, number> | null | undefined
): ChallengeDatasetStats {
  const herdIds = new Set(
    Object.values(cowMetadata)
      .map((cow) => cow.herd_id)
      .filter((herdId): herdId is number => herdId !== null && herdId !== undefined)
  );
  return {
    lactation_count: Object.keys(cowMetadata).length,
    test_day_row_count: Object.values(cowMetadata).reduce(
      (total, cow) => total + cow.dim.length,
      0
    ),
    actual_yield_count: actualYields ? Object.keys(actualYields).length : 0,
    herd_count: herdIds.size > 0 ? herdIds.size : null,
  };
}
