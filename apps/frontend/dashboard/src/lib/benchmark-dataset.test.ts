import { describe, expect, test } from "bun:test";
import { formatDatasetSources, formatDatasetStats, MODEL_LABELS } from "./benchmark-dataset";

describe("benchmark dataset helpers", () => {
  test("formats dataset stats for challenge summaries", () => {
    expect(
      formatDatasetStats({
        lactation_count: 407,
        test_day_row_count: 1628,
        actual_yield_count: 407,
      })
    ).toBe("407 lactations · 1,628 test-day rows · 407 ALY rows");
  });

  test("formats dataset sources with labels and filenames", () => {
    expect(
      formatDatasetSources([
        {
          role: "test_day_records",
          label: "Test-day records",
          filename: "TestDataSet.csv",
        },
        {
          role: "actual_yields",
          label: "Ground-truth ALY",
          filename: "ActualMilkYields.csv",
        },
      ])
    ).toBe("Test-day records: TestDataSet.csv · Ground-truth ALY: ActualMilkYields.csv");
  });

  test("uses one ISLC label across benchmark UI", () => {
    expect(MODEL_LABELS.islc).toBe("ISLC (Standard Lactation Curve interpolation)");
  });
});
