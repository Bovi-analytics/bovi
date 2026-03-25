import type { Breed } from "@/types/api";

export interface ExampleLactation {
  readonly id: string;
  readonly label: string;
  readonly description: string;
  readonly parity: number;
  readonly breed: Breed;
  readonly dim: readonly number[];
  readonly milkrecordings: readonly number[];
}

/**
 * Example datasets extracted from test data in:
 * - tests/test_data/mr_test_file.csv
 * - tests/test_data/l2_anim2_herd654.csv
 */
export const EXAMPLE_LACTATIONS: readonly ExampleLactation[] = [
  {
    id: "parity1",
    label: "Parity 1 — First Lactation",
    description: "A typical first-lactation Holstein with 8 test-day recordings.",
    parity: 1,
    breed: "H",
    dim: [27, 64, 98, 132, 174, 209, 244, 279],
    milkrecordings: [23.6, 23.1, 23.7, 24.1, 25.9, 25.4, 24.8, 22.8],
  },
  {
    id: "parity2",
    label: "Parity 2 — Second Lactation",
    description: "A second-lactation Holstein with higher peak yield and typical decline.",
    parity: 2,
    breed: "H",
    dim: [27, 61, 95, 129, 164, 199, 237, 269, 304],
    milkrecordings: [35.1, 38.8, 36.4, 35.7, 31.9, 29.3, 28.2, 27.3, 23.2],
  },
  {
    id: "parity4_high",
    label: "Parity 4 — High Yield",
    description: "A high-yielding fourth-lactation cow peaking near 50 kg/day.",
    parity: 4,
    breed: "H",
    dim: [6, 41, 83, 118, 160, 195, 230, 265, 300],
    milkrecordings: [40.1, 48.3, 41.5, 37.2, 33.8, 30.1, 28.4, 25.7, 22.9],
  },
] as const;

export const DEFAULT_LACTATION = EXAMPLE_LACTATIONS[1];
