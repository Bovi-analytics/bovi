import type { Model } from "@/types/api";

export const LACTATION_CURVE_DOCUMENTATION_URL =
  "https://github.com/Bovi-analytics/bovi/tree/main/packages/models/lactationcurve#readme";

export interface ModelParameter {
  readonly name: string;
  readonly description: string;
}

export interface ModelMetadata {
  readonly id: Model;
  readonly name: string;
  readonly formula: string;
  readonly parameterCount: number;
  readonly parameters: readonly ModelParameter[];
  readonly color: string;
  readonly description: string;
}

export const MODEL_METADATA: Record<Model, ModelMetadata> = {
  wood: {
    id: "wood",
    name: "Wood",
    formula: "y(t) = a · t^b · exp(-c · t)",
    parameterCount: 3,
    parameters: [],
    color: "#2563eb",
    description:
      "An incomplete gamma-type function combining a power-law growth term with an exponential decay term.",
  },
  wilmink: {
    id: "wilmink",
    name: "Wilmink",
    formula: "y(t) = a + b·t + c·exp(k·t)",
    parameterCount: 4,
    parameters: [],
    color: "#16a34a",
    description:
      "A combination of an exponential and a linear component. In contrast to the Fischer model, the additional parameter scaling the exponential term increases flexibility in describing early-lactation dynamics and peak formation.",
  },
  ali_schaeffer: {
    id: "ali_schaeffer",
    name: "Ali & Schaeffer",
    formula: "y(t) = a + b·(t/305) + c·(t/305)² + d·ln(305/t) + k·(ln(305/t))²",
    parameterCount: 5,
    parameters: [],
    color: "#d97706",
    description:
      "A linear regression model based on quadratic polynomials in standardized time and in the log-transformed inverse time, where the log term increases flexibility in modelling the ascending phase and peak of lactation.",
  },
  fischer: {
    id: "fischer",
    name: "Fischer",
    formula: "y(t) = a - b·t - a·exp(-c·t)",
    parameterCount: 3,
    parameters: [],
    color: "#dc2626",
    description:
      "A combination of an exponential and a linear component, characterized by an early exponential phase followed by an approximately linear decline.",
  },
  milkbot: {
    id: "milkbot",
    name: "MilkBot",
    formula: "y(t) = a · (1 - exp((c-t)/b) / 2) · exp(-d·t)",
    parameterCount: 4,
    parameters: [
      { name: "a (Scale)", description: "Overall milk production level (kg)" },
      { name: "b (Ramp)", description: "Rate of rise in early lactation" },
      { name: "c (Offset)", description: "Time correction for calving" },
      { name: "d (Decay)", description: "Rate of exponential decline" },
    ],
    color: "#9333ea",
    description:
      "An empirical, mechanistically motivated four-parameter model describing lactation as the development and decay of udder capacity. It consists of a ramp-up phase and exponential decline, with parameters controlling scale, onset, growth rate, and decay.",
  },
} as const;

/** Ordered array of all models for iteration */
export const ALL_MODELS = Object.values(MODEL_METADATA);
