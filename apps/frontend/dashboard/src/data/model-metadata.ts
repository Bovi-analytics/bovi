import type { Model } from "@/types/api";

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
    parameters: [
      { name: "a", description: "Scale factor" },
      { name: "b", description: "Rate of increase to peak" },
      { name: "c", description: "Rate of decline after peak" },
    ],
    color: "#2563eb",
    description:
      "The most widely used model due to its simplicity. " +
      "Three parameters describe the rise to peak and subsequent decline.",
  },
  wilmink: {
    id: "wilmink",
    name: "Wilmink",
    formula: "y(t) = a + b·t + c·exp(k·t)",
    parameterCount: 4,
    parameters: [
      { name: "a", description: "Base yield level" },
      { name: "b", description: "Linear decline rate" },
      { name: "c", description: "Initial rise magnitude" },
      { name: "k", description: "Decay rate (default -0.05)" },
    ],
    color: "#16a34a",
    description:
      "A linear-exponential hybrid. Combines a linear decline " +
      "with an exponential rise in early lactation.",
  },
  ali_schaeffer: {
    id: "ali_schaeffer",
    name: "Ali & Schaeffer",
    formula: "y(t) = a + b·(t/340) + c·(t/340)² + d·ln(340/t) + k·(ln(340/t))²",
    parameterCount: 5,
    parameters: [
      { name: "a", description: "Intercept" },
      { name: "b", description: "Linear coefficient" },
      { name: "c", description: "Quadratic coefficient" },
      { name: "d", description: "Logarithmic coefficient" },
      { name: "k", description: "Squared logarithmic coefficient" },
    ],
    color: "#d97706",
    description:
      "A flexible 5-parameter model using polynomial and logarithmic terms. " +
      "Best for complex lactation shapes but can overfit with few data points.",
  },
  fischer: {
    id: "fischer",
    name: "Fischer",
    formula: "y(t) = a - b·t - a·exp(-c·t)",
    parameterCount: 3,
    parameters: [
      { name: "a", description: "Asymptotic yield level" },
      { name: "b", description: "Linear decline rate" },
      { name: "c", description: "Initial rise rate" },
    ],
    color: "#dc2626",
    description:
      "A simplified exponential decay model. " +
      "Good for smooth lactation curves without sharp peaks.",
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
      "The most flexible model, describing rise, peak, and decline. " +
      "Supports both frequentist and Bayesian fitting.",
  },
} as const;

/** Ordered array of all models for iteration */
export const ALL_MODELS = Object.values(MODEL_METADATA);
