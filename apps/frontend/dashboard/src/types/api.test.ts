import { describe, expect, test } from "bun:test";
import {
  AutoencoderPredictRequestSchema,
  CharacteristicBatchRequestSchema,
  CharacteristicBatchResponseSchema,
  FittingSchema,
  MilkBotRunOptionsSchema,
  SubmissionReadSchema,
} from "./api";

describe("api schemas", () => {
  test("accepts Bayesian MilkBot fitting options", () => {
    expect(FittingSchema.parse("bayesian")).toBe("bayesian");
    expect(
      MilkBotRunOptionsSchema.parse({
        fitting: "bayesian",
        breed: "J",
        continent: "CHEN",
      })
    ).toEqual({
      fitting: "bayesian",
      breed: "J",
      continent: "CHEN",
    });
  });

  test("defaults missing submission run options for legacy responses", () => {
    const parsed = SubmissionReadSchema.parse({
      id: 1,
      challenge_id: 2,
      submission_type: "bovi_model",
      model_type: "milkbot",
      benchmark_model: "tim",
      organization: null,
      country: null,
      calculation_method: null,
      notes: null,
      user_id: null,
      stats: {
        failed_count: 0,
      },
      failed_cow_ids: [],
      created_at: null,
    });

    expect(parsed.run_options).toEqual({});
  });

  test("accepts batch characteristic requests and responses", () => {
    expect(
      CharacteristicBatchRequestSchema.parse({
        items: [
          {
            id: "wood:peak_yield",
            dim: [10, 30],
            milkrecordings: [20, 25],
            model: "wood",
            characteristic: "peak_yield",
          },
        ],
      })
    ).toEqual({
      items: [
        {
          id: "wood:peak_yield",
          dim: [10, 30],
          milkrecordings: [20, 25],
          model: "wood",
          characteristic: "peak_yield",
        },
      ],
    });

    expect(
      CharacteristicBatchResponseSchema.parse({
        results: [{ id: "wood:peak_yield", value: 30.5 }],
      })
    ).toEqual({
      results: [{ id: "wood:peak_yield", value: 30.5 }],
    });
  });

  test("accepts autoencoder periodic records", () => {
    expect(
      AutoencoderPredictRequestSchema.parse({
        dim: [10, 40, 70],
        milkrecordings: [30, 38, 35],
        parity: 2,
      })
    ).toEqual({
      dim: [10, 40, 70],
      milkrecordings: [30, 38, 35],
      parity: 2,
    });
  });

  test("rejects invalid autoencoder mixed input shapes", () => {
    expect(() =>
      AutoencoderPredictRequestSchema.parse({
        milk: [25, null, 27],
        dim: [1, 3],
        milkrecordings: [25, 27],
        parity: 2,
      })
    ).toThrow();
  });
});
