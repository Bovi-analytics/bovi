export interface HerdStatMetadata {
  readonly index: number;
  readonly name: string;
  readonly label: string;
  readonly description: string;
  readonly default: number;
}

export const HERD_STATS_METADATA: readonly HerdStatMetadata[] = [
  { index: 0, name: "Achieved21Milk", label: "21-day milk", description: "Herd average milk production in the first 21 days (kg)", default: 0.53 },
  { index: 1, name: "Achieved305Milk", label: "305-day milk", description: "Herd average 305-day total production (kg)", default: 0.50 },
  { index: 2, name: "Achieved75Milk", label: "75-day milk", description: "Herd average milk production in the first 75 days (kg)", default: 0.55 },
  { index: 3, name: "AchievedMilk", label: "Total milk", description: "Herd average total lifetime milk production (kg)", default: 0.41 },
  { index: 4, name: "DaysDry", label: "Days dry", description: "Herd average dry period length (days)", default: 0.39 },
  { index: 5, name: "DaysInMilk", label: "Days in milk", description: "Herd average current days in milk", default: 0.44 },
  { index: 6, name: "DaysOpen", label: "Open days", description: "Herd average days from calving to conception", default: 0.38 },
  { index: 7, name: "DaysPregnant", label: "Days pregnant", description: "Herd average days pregnant", default: 0.62 },
  { index: 8, name: "HistoricCalvingInterval", label: "Calving interval", description: "Herd average days between consecutive calvings", default: 0.54 },
  { index: 9, name: "QualitySequence", label: "Data quality", description: "Herd average data quality/completeness score", default: 0.33 },
] as const;

export const DEFAULT_HERD_STATS: readonly number[] = HERD_STATS_METADATA.map((s) => s.default);
