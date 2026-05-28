import { z } from "zod";

/* ------------------------------------------------------------------ */
/*  Shared enums - mirror the FastAPI Literal types exactly            */
/* ------------------------------------------------------------------ */

export const ModelSchema = z.enum(["wood", "wilmink", "ali_schaeffer", "fischer", "milkbot"]);

export const CharacteristicSchema = z.enum([
  "time_to_peak",
  "peak_yield",
  "cumulative_milk_yield",
  "persistency",
]);

export const BreedSchema = z.enum(["H", "J"]);

export const ContinentSchema = z.enum(["USA", "EU", "CHEN"]);

export const FittingSchema = z.enum(["frequentist", "bayesian"]);

export const PersistencyMethodSchema = z.enum(["derived", "literature"]);

export const ImputationMethodSchema = z.enum(["forward_fill", "backward_fill", "linear"]);

/* Inferred types - use these in components and function signatures */
export type Model = z.infer<typeof ModelSchema>;
export type Characteristic = z.infer<typeof CharacteristicSchema>;
export type Breed = z.infer<typeof BreedSchema>;
export type Continent = z.infer<typeof ContinentSchema>;
export type Fitting = z.infer<typeof FittingSchema>;
export type PersistencyMethod = z.infer<typeof PersistencyMethodSchema>;
export type ImputationMethod = z.infer<typeof ImputationMethodSchema>;

export const MilkBotRunOptionsSchema = z.object({
  fitting: FittingSchema.default("frequentist"),
  breed: BreedSchema.default("H"),
  continent: ContinentSchema.default("USA"),
});
export type MilkBotRunOptions = z.infer<typeof MilkBotRunOptionsSchema>;

/* ------------------------------------------------------------------ */
/*  Request schemas                                                    */
/* ------------------------------------------------------------------ */

export const FitRequestSchema = z.object({
  dim: z.array(z.number().int()),
  milkrecordings: z.array(z.number()),
  model: ModelSchema.optional(),
  fitting: FittingSchema.optional(),
  breed: BreedSchema.optional(),
  parity: z.number().int().min(1).optional(),
  continent: ContinentSchema.optional(),
});

export const CharacteristicRequestSchema = z.object({
  dim: z.array(z.number().int()),
  milkrecordings: z.array(z.number()),
  model: ModelSchema.optional(),
  characteristic: CharacteristicSchema.optional(),
  fitting: FittingSchema.optional(),
  breed: BreedSchema.optional(),
  parity: z.number().int().min(1).optional(),
  continent: ContinentSchema.optional(),
  persistency_method: PersistencyMethodSchema.optional(),
  lactation_length: z.number().int().min(1).optional(),
});

export const CharacteristicBatchItemSchema = CharacteristicRequestSchema.extend({
  id: z.union([z.number(), z.string()]).optional(),
});

export const CharacteristicBatchRequestSchema = z.object({
  items: z.array(CharacteristicBatchItemSchema).min(1),
});

export const PredictRequestSchema = z.object({
  t: z.array(z.number().int()),
  a: z.number(),
  b: z.number(),
  c: z.number(),
  d: z.number(),
});

export const YieldEstimateRequestSchema = z.object({
  dim: z.array(z.number().int()),
  milkrecordings: z.array(z.number()),
  test_ids: z.array(z.union([z.number(), z.string()])).optional(),
});
export const TestIntervalRequestSchema = YieldEstimateRequestSchema;

export const AutoencoderPredictRequestSchema = z
  .object({
    milk: z.array(z.number().nullable()).min(1).optional(),
    dim: z.array(z.number().int().min(1).max(304)).min(1).optional(),
    milkrecordings: z.array(z.number()).min(1).optional(),
    events: z.array(z.string()).optional(),
    parity: z.number().int().min(1).max(12),
    herd_id: z.number().int().optional(),
    herd_stats: z.array(z.number()).length(10).optional(),
    imputation_method: ImputationMethodSchema.optional(),
  })
  .superRefine((value, ctx) => {
    const hasDaily = value.milk !== undefined;
    const hasPeriodic = value.dim !== undefined || value.milkrecordings !== undefined;
    if (hasDaily && hasPeriodic) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Provide either milk or dim + milkrecordings, not both.",
      });
    }
    if (!hasDaily && !hasPeriodic) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Provide either milk or dim + milkrecordings.",
      });
    }
    if (hasPeriodic && (value.dim === undefined || value.milkrecordings === undefined)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Periodic input requires both dim and milkrecordings.",
      });
    }
    if (
      value.dim !== undefined &&
      value.milkrecordings !== undefined &&
      value.dim.length !== value.milkrecordings.length
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "dim and milkrecordings must have the same length.",
      });
    }
  });

/* ------------------------------------------------------------------ */
/*  Response schemas                                                   */
/* ------------------------------------------------------------------ */

export const FitResponseSchema = z.object({
  predictions: z.array(z.number()),
});

export const CharacteristicResponseSchema = z.object({
  value: z.number().nullable(),
});

export const CharacteristicBatchResultSchema = z.object({
  id: z.union([z.number(), z.string()]).nullable().optional(),
  value: z.number().nullable(),
});

export const CharacteristicBatchResponseSchema = z.object({
  results: z.array(CharacteristicBatchResultSchema),
});

export const PredictResponseSchema = z.object({
  predictions: z.array(z.number()),
});

export const AutoencoderPredictResponseSchema = z.object({
  predictions: z.array(z.number()),
  latent_vector: z.array(z.number()).nullable(),
});

export const YieldEstimateResultSchema = z.object({
  test_id: z.union([z.number(), z.string()]),
  total_305_yield: z.number(),
});
export const TestIntervalResultSchema = YieldEstimateResultSchema;

export const YieldEstimateResponseSchema = z.object({
  results: z.array(YieldEstimateResultSchema),
});
export const TestIntervalResponseSchema = YieldEstimateResponseSchema;

/* ------------------------------------------------------------------ */
/*  Inferred request/response types                                    */
/* ------------------------------------------------------------------ */

export type FitRequest = z.infer<typeof FitRequestSchema>;
export type FitResponse = z.infer<typeof FitResponseSchema>;
export type CharacteristicRequest = z.infer<typeof CharacteristicRequestSchema>;
export type CharacteristicResponse = z.infer<typeof CharacteristicResponseSchema>;
export type CharacteristicBatchItem = z.infer<typeof CharacteristicBatchItemSchema>;
export type CharacteristicBatchRequest = z.infer<typeof CharacteristicBatchRequestSchema>;
export type CharacteristicBatchResponse = z.infer<typeof CharacteristicBatchResponseSchema>;
export type PredictRequest = z.infer<typeof PredictRequestSchema>;
export type PredictResponse = z.infer<typeof PredictResponseSchema>;
export type AutoencoderPredictRequest = z.infer<typeof AutoencoderPredictRequestSchema>;
export type AutoencoderPredictResponse = z.infer<typeof AutoencoderPredictResponseSchema>;
export type YieldEstimateRequest = z.infer<typeof YieldEstimateRequestSchema>;
export type YieldEstimateResponse = z.infer<typeof YieldEstimateResponseSchema>;
export type TestIntervalRequest = z.infer<typeof TestIntervalRequestSchema>;
export type TestIntervalResponse = z.infer<typeof TestIntervalResponseSchema>;

/* ------------------------------------------------------------------ */
/*  Herd profile schemas                                               */
/* ------------------------------------------------------------------ */

export const HerdProfileSchema = z.object({
  id: z.number(),
  user_id: z.number().nullable().optional(),
  organization_id: z.number().nullable().optional(),
  name: z.string().max(100),
  description: z.string().max(500),
  achieved_21_milk: z.number().min(0).max(1),
  achieved_305_milk: z.number().min(0).max(1),
  achieved_75_milk: z.number().min(0).max(1),
  achieved_milk: z.number().min(0).max(1),
  days_dry: z.number().min(0).max(1),
  days_in_milk: z.number().min(0).max(1),
  days_open: z.number().min(0).max(1),
  days_pregnant: z.number().min(0).max(1),
  historic_calving_interval: z.number().min(0).max(1),
  quality_sequence: z.number().min(0).max(1),
  created_at: z.string().nullable(),
  updated_at: z.string().nullable(),
});

export const HerdProfileCreateSchema = HerdProfileSchema.omit({
  id: true,
  user_id: true,
  created_at: true,
  updated_at: true,
}).extend({ organization_id: z.number() });

export const HerdProfileListSchema = z.array(HerdProfileSchema);

export type HerdProfile = z.infer<typeof HerdProfileSchema>;
export type HerdProfileCreate = z.infer<typeof HerdProfileCreateSchema>;

export const CowRecordSchema = z.object({
  cow_id: z.string(),
  parity: z.number().nullable(),
  dim: z.array(z.number()),
  milk_kg: z.array(z.number()),
});

export type CowRecord = z.infer<typeof CowRecordSchema>;

export const HerdProfileUploadResponseSchema = z.object({
  upload_id: z.string().nullable().optional(),
  stats: z.record(z.string(), z.number()),
  raw_stats: z.record(z.string(), z.number()),
  format_detected: z.enum(["aggregated", "icar_test_day"]),
  row_count: z.number(),
  warnings: z.array(z.string()),
  cow_count: z.number().nullable().optional(),
  detected_parity: z.number().nullable().optional(),
  columns: z.array(z.string()).default([]),
  column_mapping: z.record(z.string(), z.string()).default({}),
  mapping_required: z.boolean().default(false),
  cows: z.array(CowRecordSchema).default([]),
});

export type HerdProfileUploadResponse = z.infer<typeof HerdProfileUploadResponseSchema>;

/* ------------------------------------------------------------------ */
/*  Preset cow-dataset schemas                                         */
/* ------------------------------------------------------------------ */

export const PresetDatasetKeySchema = z.enum(["aurora", "sunnyside"]);
export const PresetSizeKeySchema = z.enum(["small", "medium", "large"]);
export const PresetPeriodKeySchema = z.enum(["recent", "old", "mixed"]);

export const PresetCowSchema = z.object({
  cow_id: z.string(),
  display_name: z.string(),
  parity: z.number().nullable(),
  herd_id: z.number().int().nullable().optional(),
  dim: z.array(z.number()),
  milk_kg: z.array(z.number()),
});

export const PresetDatasetResponseSchema = z.object({
  dataset: z.string(),
  size: z.string(),
  period: z.string(),
  cow_count: z.number(),
  cows: z.array(PresetCowSchema),
});

export const PresetCountsResponseSchema = z.object({
  dataset: z.string(),
  counts: z.record(PresetPeriodKeySchema, z.record(PresetSizeKeySchema, z.number())),
});

export const PresetHerdStatsResponseSchema = z.object({
  dataset: z.string(),
  size: z.string(),
  period: z.string(),
  parity: z.number().nullable(),
  cow_count: z.number(),
  raw_stats: z.record(z.string(), z.number()),
  stats: z.record(z.string(), z.number()),
  warnings: z.array(z.string()),
});

export type PresetDatasetKey = z.infer<typeof PresetDatasetKeySchema>;
export type PresetSizeKey = z.infer<typeof PresetSizeKeySchema>;
export type PresetPeriodKey = z.infer<typeof PresetPeriodKeySchema>;
export type PresetCow = z.infer<typeof PresetCowSchema>;
export type PresetDatasetResponse = z.infer<typeof PresetDatasetResponseSchema>;
export type PresetCountsResponse = z.infer<typeof PresetCountsResponseSchema>;
export type PresetHerdStatsResponse = z.infer<typeof PresetHerdStatsResponseSchema>;

/* ------------------------------------------------------------------ */
/*  Benchmark - Challenges                                             */
/* ------------------------------------------------------------------ */

export const ChallengeDatasetSourceSchema = z.object({
  role: z.string(),
  label: z.string(),
  filename: z.string().nullable().optional(),
});
export type ChallengeDatasetSource = z.infer<typeof ChallengeDatasetSourceSchema>;

export const ChallengeDatasetStatsSchema = z.object({
  lactation_count: z.number().nullable().optional(),
  test_day_row_count: z.number().nullable().optional(),
  actual_yield_count: z.number().nullable().optional(),
  herd_count: z.number().nullable().optional(),
});
export type ChallengeDatasetStats = z.infer<typeof ChallengeDatasetStatsSchema>;

export const ChallengeReadSchema = z.object({
  id: z.number(),
  dataset: z.string(),
  size: z.string(),
  period: z.string(),
  name: z.string().nullable().optional(),
  source: z.string().nullable().optional(),
  user_id: z.number().nullable(),
  organization_id: z.number().nullable().optional(),
  created_at: z.string().nullable(),
  row_count: z.number().nullable().optional(),
  cow_count: z.number().nullable().optional(),
  actual_yield_count: z.number().nullable().optional(),
  ingest_status: z.string().optional(),
  dataset_sources: z.preprocess((value) => value ?? [], z.array(ChallengeDatasetSourceSchema)),
  dataset_stats: z.preprocess((value) => value ?? {}, ChallengeDatasetStatsSchema),
});
export type ChallengeRead = z.infer<typeof ChallengeReadSchema>;

const BenchmarkCowMetadataSchema = z.object({
  parity: z.number().nullable().optional(),
  herd_id: z.number().nullable().optional(),
  dim: z.array(z.number()),
  milk_kg: z.array(z.number()),
});

export const ChallengeDetailSchema = ChallengeReadSchema.extend({
  cow_metadata: z.record(z.string(), BenchmarkCowMetadataSchema),
  actual_yields: z.record(z.string(), z.number()).nullable().optional(),
});
export type ChallengeDetail = z.infer<typeof ChallengeDetailSchema>;

export const ChallengeListSchema = z.array(ChallengeReadSchema);

export const ChallengeCreatePresetSchema = z.object({
  source: z.literal("preset").default("preset"),
  preset: z.literal("icar").default("icar"),
  name: z.string().optional(),
  organization_id: z.number(),
});
export type ChallengeCreatePreset = z.infer<typeof ChallengeCreatePresetSchema>;

/* ------------------------------------------------------------------ */
/*  Benchmark - Submissions                                            */
/* ------------------------------------------------------------------ */

export const ParityStatsSchema = z.object({
  pearson: z.number().nullable(),
  rmse: z.number().nullable(),
  mae: z.number().nullable(),
  mape: z.number().nullable(),
  n: z.number(),
});

export const VsBlockSchema = z.object({
  overall: ParityStatsSchema,
  by_parity: z.record(z.string(), ParityStatsSchema),
});
export type ParityStats = z.infer<typeof ParityStatsSchema>;
export type VsBlock = z.infer<typeof VsBlockSchema>;

export const ComparisonStatsSchema = z.object({
  version: z.number().optional(),
  challenger_vs_aly: VsBlockSchema.optional(),
  benchmark_vs_aly: VsBlockSchema.optional(),
  challenger_vs_benchmark: VsBlockSchema.optional(),
  failed_count: z.number(),
  // legacy fields (v1)
  overall: ParityStatsSchema.optional(),
  by_parity: z.record(z.string(), ParityStatsSchema).optional(),
  vs_aly: VsBlockSchema.optional(),
});
export type ComparisonStats = z.infer<typeof ComparisonStatsSchema>;

const RunOptionsSchema = z
  .union([z.record(z.string(), z.unknown()), z.null(), z.undefined()])
  .transform((value): Record<string, unknown> => value ?? {});

export const SubmissionReadSchema = z.object({
  id: z.number(),
  challenge_id: z.number(),
  submission_type: z.string(),
  model_type: z.string().nullable(),
  benchmark_model: z.string().nullable().optional(),
  run_options: RunOptionsSchema,
  organization: z.string().nullable(),
  country: z.string().nullable(),
  calculation_method: z.string().nullable(),
  notes: z.string().nullable(),
  user_id: z.number().nullable(),
  organization_id: z.number().nullable().optional(),
  stats: ComparisonStatsSchema,
  failed_cow_ids: z.array(z.string()),
  created_at: z.string().nullable(),
  row_count: z.number().nullable().optional(),
  submitted_yield_count: z.number().nullable().optional(),
  benchmark_yield_count: z.number().nullable().optional(),
  failed_count: z.number().optional(),
  ingest_status: z.string().optional(),
});
export type SubmissionRead = z.infer<typeof SubmissionReadSchema>;

export const BenchmarkModelSchema = z.enum([
  "wood",
  "wilmink",
  "ali_schaeffer",
  "fischer",
  "milkbot",
  "autoencoder",
  "tim",
  "islc",
  "best_predict",
]);
export type BenchmarkModel = z.infer<typeof BenchmarkModelSchema>;

export const SubmissionListSchema = z.array(SubmissionReadSchema);

/* ------------------------------------------------------------------ */
/*  Admin overview                                                     */
/* ------------------------------------------------------------------ */

export const AdminDataCategorySchema = z.enum([
  "benchmark_submission",
  "benchmark_challenge",
  "herd_dataset_upload",
  "herd_profile",
]);
export type AdminDataCategory = z.infer<typeof AdminDataCategorySchema>;

export const AdminOverviewItemSchema = z.object({
  item_type: AdminDataCategorySchema,
  item_type_label: z.string(),
  id: z.string(),
  numeric_id: z.number().nullable().optional(),
  challenge_id: z.number().nullable().optional(),
  organization_id: z.number().nullable().optional(),
  organization_name: z.string().nullable().optional(),
  user_id: z.number().nullable().optional(),
  user_email: z.string().nullable().optional(),
  user_name: z.string().nullable().optional(),
  title: z.string(),
  created_at: z.string().nullable(),
  status: z.string(),
  source: z.string().nullable().optional(),
  submission_type: z.string().nullable().optional(),
  benchmark_model: z.string().nullable().optional(),
  row_count: z.number().nullable().optional(),
  cow_count: z.number().nullable().optional(),
  failed_count: z.number(),
  primary_metric_label: z.string().nullable().optional(),
  primary_metric_value: z.number().nullable().optional(),
});
export type AdminOverviewItem = z.infer<typeof AdminOverviewItemSchema>;

export const AdminOrganizationBreakdownSchema = z.object({
  organization_id: z.number().nullable().optional(),
  organization_name: z.string(),
  user_count: z.number(),
  total_items: z.number(),
  benchmark_submissions: z.number(),
  benchmark_challenges: z.number(),
  herd_dataset_uploads: z.number(),
  herd_profiles: z.number(),
  failed_items: z.number(),
  latest_activity_at: z.string().nullable(),
});
export type AdminOrganizationBreakdown = z.infer<typeof AdminOrganizationBreakdownSchema>;

export const AdminCategoryBreakdownSchema = z.object({
  item_type: AdminDataCategorySchema,
  label: z.string(),
  count: z.number(),
  failed_count: z.number(),
  latest_activity_at: z.string().nullable(),
});
export type AdminCategoryBreakdown = z.infer<typeof AdminCategoryBreakdownSchema>;

export const AdminOverviewKpisSchema = z.object({
  total_items: z.number(),
  organizations: z.number(),
  users: z.number(),
  benchmark_submissions: z.number(),
  benchmark_challenges: z.number(),
  herd_dataset_uploads: z.number(),
  herd_profiles: z.number(),
  failed_items: z.number(),
  latest_activity_at: z.string().nullable(),
});
export type AdminOverviewKpis = z.infer<typeof AdminOverviewKpisSchema>;

export const AdminOverviewResponseSchema = z.object({
  kpis: AdminOverviewKpisSchema,
  by_organization: z.array(AdminOrganizationBreakdownSchema),
  by_category: z.array(AdminCategoryBreakdownSchema),
  items: z.array(AdminOverviewItemSchema),
});
export type AdminOverviewResponse = z.infer<typeof AdminOverviewResponseSchema>;
