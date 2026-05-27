import { describe, expect, test } from "bun:test";
import { EXAMPLE_AUTOENCODER_DATA } from "../data/example-autoencoder";
import {
  DAILY_MODEL_INPUT_DAYS,
  prepareDailyModelInput,
  prepareObservedDailyModelInput,
} from "./daily-model-input";

describe("prepareDailyModelInput", () => {
  test("uses the full daily model horizon instead of sampled test-day points", () => {
    const partialWithGaps = EXAMPLE_AUTOENCODER_DATA.find((data) => data.id === "partial-with-gaps");

    expect(partialWithGaps).toBeDefined();

    const result = prepareDailyModelInput(partialWithGaps!.milk, {
      useImputation: false,
      imputationMethod: "forward_fill",
    });

    expect(result.dim).toHaveLength(DAILY_MODEL_INPUT_DAYS);
    expect(result.dim).toEqual(Array.from({ length: DAILY_MODEL_INPUT_DAYS }, (_, index) => index + 1));
    expect(result.milk).toHaveLength(DAILY_MODEL_INPUT_DAYS);
    expect(result.milk).not.toHaveLength(7);
  });

  test("can include all source days when a longer horizon is requested", () => {
    const partialWithGaps = EXAMPLE_AUTOENCODER_DATA.find((data) => data.id === "partial-with-gaps");

    expect(partialWithGaps).toBeDefined();

    const observedCount = partialWithGaps!.milk.filter((value) => value !== null).length;
    const result = prepareDailyModelInput(partialWithGaps!.milk, {
      useImputation: false,
      imputationMethod: "forward_fill",
      maxDays: partialWithGaps!.milk.length,
    });

    expect(observedCount).toBe(200);
    expect(result.dim).toHaveLength(partialWithGaps!.milk.length);
    expect(result.milk.filter((value) => value !== 0)).toHaveLength(observedCount);
  });

  test("uses zero fill when imputation is disabled", () => {
    const result = prepareDailyModelInput([12, null, 18, null], {
      useImputation: false,
      imputationMethod: "linear",
    });

    expect(result.milk).toEqual([12, 0, 18, 0]);
    expect(result.missingCount).toBe(2);
  });

  test("forward fills missing values when imputation is enabled", () => {
    const result = prepareDailyModelInput([null, 10, null, 14, null], {
      useImputation: true,
      imputationMethod: "forward_fill",
    });

    expect(result.milk).toEqual([10, 10, 10, 14, 14]);
  });

  test("backward fills missing values when imputation is enabled", () => {
    const result = prepareDailyModelInput([null, 10, null, 14, null], {
      useImputation: true,
      imputationMethod: "backward_fill",
    });

    expect(result.milk).toEqual([10, 10, 14, 14, 14]);
  });

  test("linearly interpolates missing values when imputation is enabled", () => {
    const result = prepareDailyModelInput([null, 10, null, null, 16, null], {
      useImputation: true,
      imputationMethod: "linear",
    });

    expect(result.milk).toEqual([10, 10, 12, 14, 16, 16]);
  });

  test("keeps only observed values for classical daily model input", () => {
    const result = prepareObservedDailyModelInput([12, null, 18, null, 20]);

    expect(result.dim).toEqual([1, 3, 5]);
    expect(result.milk).toEqual([12, 18, 20]);
    expect(result.milk).not.toContain(0);
    expect(result.missingCount).toBe(2);
  });

  test("does not fabricate zero yields when all daily values are missing", () => {
    const result = prepareObservedDailyModelInput([null, null, null]);

    expect(result.dim).toEqual([]);
    expect(result.milk).toEqual([]);
    expect(result.missingCount).toBe(3);
  });
});
