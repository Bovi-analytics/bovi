import { describe, expect, test } from "bun:test";
import {
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
});
