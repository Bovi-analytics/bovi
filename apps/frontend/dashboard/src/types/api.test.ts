import { describe, expect, test } from "bun:test";
import {
  AutoencoderPredictRequestSchema,
  AdminOverviewResponseSchema,
  CharacteristicBatchRequestSchema,
  CharacteristicBatchResponseSchema,
  ChallengeReadSchema,
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

  test("defaults missing challenge dataset metadata for legacy responses", () => {
    const parsed = ChallengeReadSchema.parse({
      id: 1,
      dataset: "icar",
      size: "full",
      period: "all",
      name: "Demo dataset",
      source: "preset",
      user_id: null,
      created_at: null,
    });

    expect(parsed.dataset_sources).toEqual([]);
    expect(parsed.dataset_stats).toEqual({});
  });

  test("accepts challenge dataset sources and stats", () => {
    const parsed = ChallengeReadSchema.parse({
      id: 1,
      dataset: "upload",
      size: "custom",
      period: "custom",
      name: "Uploaded",
      source: "upload",
      user_id: null,
      created_at: null,
      dataset_sources: [
        {
          role: "test_day_records",
          label: "Test-day records",
          filename: "test_day.csv",
        },
      ],
      dataset_stats: {
        lactation_count: 2,
        test_day_row_count: 8,
        actual_yield_count: 2,
        herd_count: null,
      },
    });

    expect(parsed.dataset_sources[0]?.filename).toBe("test_day.csv");
    expect(parsed.dataset_stats.test_day_row_count).toBe(8);
  });

  test("accepts admin overview records for all submission categories", () => {
    const parsed = AdminOverviewResponseSchema.parse({
      kpis: {
        total_items: 4,
        organizations: 2,
        users: 2,
        benchmark_submissions: 1,
        benchmark_challenges: 1,
        herd_dataset_uploads: 1,
        herd_profiles: 1,
        failed_items: 1,
        latest_activity_at: "2026-05-28T10:00:00Z",
      },
      by_organization: [
        {
          organization_id: 1,
          organization_name: "Dairy One",
          user_count: 1,
          total_items: 2,
          benchmark_submissions: 1,
          benchmark_challenges: 1,
          herd_dataset_uploads: 0,
          herd_profiles: 0,
          failed_items: 1,
          latest_activity_at: "2026-05-28T10:00:00Z",
        },
      ],
      by_category: [
        {
          item_type: "benchmark_submission",
          label: "Benchmark submissions",
          count: 1,
          failed_count: 1,
          latest_activity_at: "2026-05-28T10:00:00Z",
        },
        {
          item_type: "benchmark_challenge",
          label: "Benchmark challenges",
          count: 1,
          failed_count: 0,
          latest_activity_at: null,
        },
        {
          item_type: "herd_dataset_upload",
          label: "Herd dataset uploads",
          count: 1,
          failed_count: 0,
          latest_activity_at: null,
        },
        {
          item_type: "herd_profile",
          label: "Herd profiles",
          count: 1,
          failed_count: 0,
          latest_activity_at: null,
        },
      ],
      items: [
        {
          item_type: "benchmark_submission",
          item_type_label: "Benchmark submissions",
          id: "7",
          numeric_id: 7,
          challenge_id: 3,
          organization_id: 1,
          organization_name: "Dairy One",
          user_id: 1,
          user_email: "admin@example.test",
          user_name: "Admin",
          title: "Uploaded method",
          created_at: "2026-05-28T10:00:00Z",
          status: "ready",
          source: "benchmark",
          submission_type: "own_method",
          benchmark_model: "tim",
          row_count: 100,
          cow_count: 25,
          failed_count: 1,
          primary_metric_label: "RMSE",
          primary_metric_value: 12.3,
        },
      ],
    });

    expect(parsed.items[0]?.item_type).toBe("benchmark_submission");
    expect(parsed.kpis.failed_items).toBe(1);
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
