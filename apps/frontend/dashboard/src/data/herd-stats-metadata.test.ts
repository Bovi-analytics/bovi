import { describe, expect, test } from "bun:test";
import { HERD_STATS_METADATA } from "./herd-stats-metadata";

const PACKAGE_RANGES: Record<string, readonly [number, number]> = {
  Achieved21Milk: [0, 50],
  Achieved305Milk: [3000, 15000],
  Achieved75Milk: [0, 50],
  AchievedMilk: [3000, 20000],
  DaysDry: [0, 150],
  DaysInMilk: [0, 600],
  DaysOpen: [0, 300],
  DaysPregnant: [0, 283],
  HistoricCalvingInterval: [300, 600],
  QualitySequence: [0, 1],
};

describe("herd stats metadata", () => {
  test("uses the package herd stats normalization ranges", () => {
    for (const stat of HERD_STATS_METADATA) {
      expect([stat.rawMin, stat.rawMax]).toEqual(PACKAGE_RANGES[stat.name]);
    }
  });
});
