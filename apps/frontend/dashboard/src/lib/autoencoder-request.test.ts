import { describe, expect, test } from "bun:test";
import { buildAutoencoderPredictRequest } from "./autoencoder-request";

describe("buildAutoencoderPredictRequest", () => {
  test("always includes parity", () => {
    const request = buildAutoencoderPredictRequest({
      milk: [24, null, 28],
      parity: 2,
    });

    expect(request).toEqual({
      milk: [24, null, 28],
      parity: 2,
    });
  });

  test("can build a periodic-record request", () => {
    const request = buildAutoencoderPredictRequest({
      dim: [10, 40, 70],
      milkrecordings: [30, 38, 35],
      parity: 2,
    });

    expect(request).toEqual({
      dim: [10, 40, 70],
      milkrecordings: [30, 38, 35],
      parity: 2,
    });
  });

  test("includes herd_id only when a real herdId is available", () => {
    const withHerdId = buildAutoencoderPredictRequest({
      milk: [24],
      parity: 2,
      herdId: 2942694,
    });
    const withoutHerdId = buildAutoencoderPredictRequest({
      milk: [24],
      parity: 2,
    });

    expect(withHerdId.herd_id).toBe(2942694);
    expect(withoutHerdId).not.toHaveProperty("herd_id");
  });

  test("preserves explicit imputation method and herd stats", () => {
    const request = buildAutoencoderPredictRequest({
      milk: [24, 0, 28],
      parity: 3,
      events: ["calving", "pad", "pad"],
      herdStats: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
      imputationMethod: "linear",
    });

    expect(request).toEqual({
      milk: [24, 0, 28],
      parity: 3,
      events: ["calving", "pad", "pad"],
      herd_stats: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
      imputation_method: "linear",
    });
  });
});
