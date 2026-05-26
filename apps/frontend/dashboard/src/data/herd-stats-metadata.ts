export interface HerdStatMetadata {
  readonly index: number;
  readonly name: string;
  readonly label: string;
  readonly description: string;
  readonly default: number;
  readonly unit: string;
  readonly rawMin: number;
  readonly rawMax: number;
}

export const HERD_STATS_METADATA: readonly HerdStatMetadata[] = [
  {
    index: 0,
    name: "Achieved21Milk",
    label: "21-day milk",
    description: "Herd average milk production in the first 21 days (kg)",
    default: 0.53,
    unit: "kg",
    rawMin: 76,
    rawMax: 1216,
  },
  {
    index: 1,
    name: "Achieved305Milk",
    label: "305-day milk",
    description: "Herd average 305-day total production (kg)",
    default: 0.5,
    unit: "kg",
    rawMin: 2925,
    rawMax: 15000,
  },
  {
    index: 2,
    name: "Achieved75Milk",
    label: "75-day milk",
    description: "Herd average milk production in the first 75 days (kg)",
    default: 0.55,
    unit: "kg",
    rawMin: 609,
    rawMax: 4729,
  },
  {
    index: 3,
    name: "AchievedMilk",
    label: "Total milk",
    description: "Herd average total lifetime milk production (kg)",
    default: 0.41,
    unit: "kg",
    rawMin: 100,
    rawMax: 16998,
  },
  {
    index: 4,
    name: "DaysDry",
    label: "Days dry",
    description: "Herd average dry period length (days)",
    default: 0.39,
    unit: "days",
    rawMin: 30,
    rawMax: 120,
  },
  {
    index: 5,
    name: "DaysInMilk",
    label: "Days in milk",
    description: "Herd average current days in milk",
    default: 0.44,
    unit: "days",
    rawMin: 103,
    rawMax: 450,
  },
  {
    index: 6,
    name: "DaysOpen",
    label: "Open days",
    description: "Herd average days from calving to conception",
    default: 0.38,
    unit: "days",
    rawMin: 60,
    rawMax: 200,
  },
  {
    index: 7,
    name: "DaysPregnant",
    label: "Days pregnant",
    description: "Herd average days pregnant",
    default: 0.62,
    unit: "days",
    rawMin: 0,
    rawMax: 289,
  },
  {
    index: 8,
    name: "HistoricCalvingInterval",
    label: "Calving interval",
    description: "Herd average days between consecutive calvings",
    default: 0.54,
    unit: "days",
    rawMin: 330,
    rawMax: 450,
  },
  {
    index: 9,
    name: "QualitySequence",
    label: "Data quality",
    description: "Herd average data quality/completeness score",
    default: 1,
    unit: "",
    rawMin: 0,
    rawMax: 1,
  },
] as const;

export const QUALITY_SEQUENCE_INDEX = 9;
export const QUALITY_SEQUENCE_VALUE = 1;
export const VISIBLE_HERD_STATS_METADATA: readonly HerdStatMetadata[] = HERD_STATS_METADATA.filter(
  (s) => s.index !== QUALITY_SEQUENCE_INDEX
);

export const DEFAULT_HERD_STATS: readonly number[] = HERD_STATS_METADATA.map((s) => s.default);

/** Convert a normalized [0, 1] value to its raw unit value. */
export function toRaw(stat: HerdStatMetadata, normalized: number): number {
  return normalized * (stat.rawMax - stat.rawMin) + stat.rawMin;
}

/** Convert a raw unit value back to normalized [0, 1]. */
export function toNormalized(stat: HerdStatMetadata, raw: number): number {
  if (stat.rawMax === stat.rawMin) return 0;
  return (raw - stat.rawMin) / (stat.rawMax - stat.rawMin);
}
