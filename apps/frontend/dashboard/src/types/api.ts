import { z } from "zod";

/* ------------------------------------------------------------------ */
/*  Shared enums — mirror the FastAPI Literal types exactly            */
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

export const FittingSchema = z.enum(["frequentist"]);

export const PersistencyMethodSchema = z.enum(["derived", "literature"]);

/* Inferred types — use these in components and function signatures */
export type Model = z.infer<typeof ModelSchema>;
export type Characteristic = z.infer<typeof CharacteristicSchema>;
export type Breed = z.infer<typeof BreedSchema>;
export type Continent = z.infer<typeof ContinentSchema>;
export type Fitting = z.infer<typeof FittingSchema>;
export type PersistencyMethod = z.infer<typeof PersistencyMethodSchema>;

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

export const PredictRequestSchema = z.object({
  t: z.array(z.number().int()),
  a: z.number(),
  b: z.number(),
  c: z.number(),
  d: z.number(),
});

export const TestIntervalRequestSchema = z.object({
  dim: z.array(z.number().int()),
  milkrecordings: z.array(z.number()),
  test_ids: z.array(z.union([z.number(), z.string()])).optional(),
});

/* ------------------------------------------------------------------ */
/*  Response schemas                                                   */
/* ------------------------------------------------------------------ */

export const FitResponseSchema = z.object({
  predictions: z.array(z.number()),
});

export const CharacteristicResponseSchema = z.object({
  value: z.number(),
});

export const PredictResponseSchema = z.object({
  predictions: z.array(z.number()),
});

export const TestIntervalResultSchema = z.object({
  test_id: z.union([z.number(), z.string()]),
  total_305_yield: z.number(),
});

export const TestIntervalResponseSchema = z.object({
  results: z.array(TestIntervalResultSchema),
});

/* ------------------------------------------------------------------ */
/*  Inferred request/response types                                    */
/* ------------------------------------------------------------------ */

export type FitRequest = z.infer<typeof FitRequestSchema>;
export type FitResponse = z.infer<typeof FitResponseSchema>;
export type CharacteristicRequest = z.infer<typeof CharacteristicRequestSchema>;
export type CharacteristicResponse = z.infer<typeof CharacteristicResponseSchema>;
export type PredictRequest = z.infer<typeof PredictRequestSchema>;
export type PredictResponse = z.infer<typeof PredictResponseSchema>;
export type TestIntervalRequest = z.infer<typeof TestIntervalRequestSchema>;
export type TestIntervalResponse = z.infer<typeof TestIntervalResponseSchema>;
