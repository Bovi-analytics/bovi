/**
 * Utilities for mapping between HerdProfile DB fields (snake_case) and the
 * number[] expected by HerdStatsForm (indexed by HERD_STATS_METADATA order).
 */

import type { HerdProfile, HerdProfileCreate } from "@/types/api";

type StatField = keyof Omit<HerdProfileCreate, "name" | "description">;

/**
 * snake_case field names in the same order as HERD_STATS_METADATA indices 0–9.
 * Index 0 = Achieved21Milk, index 9 = QualitySequence.
 */
export const HERD_PROFILE_FIELD_ORDER: StatField[] = [
  "achieved_21_milk",
  "achieved_305_milk",
  "achieved_75_milk",
  "achieved_milk",
  "days_dry",
  "days_in_milk",
  "days_open",
  "days_pregnant",
  "historic_calving_interval",
  "quality_sequence",
];

/** Convert a saved HerdProfile → number[] for HerdStatsForm. */
export function herdProfileToStats(profile: HerdProfile): number[] {
  return HERD_PROFILE_FIELD_ORDER.map((field) => profile[field] as number);
}

/** Convert number[] from HerdStatsForm → stat fields for HerdProfileCreate. */
export function statsToHerdProfileFields(stats: number[]): Record<StatField, number> {
  return Object.fromEntries(
    HERD_PROFILE_FIELD_ORDER.map((field, i) => [field, stats[i] ?? 0])
  ) as Record<StatField, number>;
}
