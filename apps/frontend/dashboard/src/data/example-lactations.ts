import type { Breed } from "@/types/api";

export type ExampleLactationSource = "synthetic" | "icar";

export interface ExampleLactation {
  readonly id: string;
  readonly label: string;
  readonly description: string;
  readonly parity: number;
  readonly breed: Breed;
  readonly dim: readonly number[];
  readonly milkrecordings: readonly number[];
  readonly source: ExampleLactationSource;
}

/**
 * Example datasets.
 *
 * - `source: "synthetic"` - illustrative lactations, hand-crafted for teaching purposes.
 * - `source: "icar"` - anonymized reference cows with real test-day records in kg.
 */
export const EXAMPLE_LACTATIONS: readonly ExampleLactation[] = [
  {
    id: "parity1",
    label: "Parity 1 - First lactation",
    description: "A typical first-lactation Holstein with 8 test-day recordings.",
    parity: 1,
    breed: "H",
    dim: [27, 64, 98, 132, 174, 209, 244, 279],
    milkrecordings: [23.6, 23.1, 23.7, 24.1, 25.9, 25.4, 24.8, 22.8],
    source: "synthetic",
  },
  {
    id: "parity2",
    label: "Parity 2 - Second lactation",
    description: "A second-lactation Holstein with higher peak yield and typical decline.",
    parity: 2,
    breed: "H",
    dim: [27, 61, 95, 129, 164, 199, 237, 269, 304],
    milkrecordings: [35.1, 38.8, 36.4, 35.7, 31.9, 29.3, 28.2, 27.3, 23.2],
    source: "synthetic",
  },
  {
    id: "parity4_high",
    label: "Parity 4 - High yield",
    description: "A high-yielding fourth-lactation cow peaking near 50 kg/day.",
    parity: 4,
    breed: "H",
    dim: [6, 41, 83, 118, 160, 195, 230, 265, 300],
    milkrecordings: [40.1, 48.3, 41.5, 37.2, 33.8, 30.1, 28.4, 25.7, 22.9],
    source: "synthetic",
  },
  {
    id: "icar-2348",
    label: "Reference cow 2348 - parity 1",
    description:
      "Anonymized test-day records: first-lactation heifer, 11 measurements, peak ~34 kg.",
    parity: 1,
    breed: "H",
    dim: [7, 35, 63, 91, 119, 147, 175, 203, 231, 259, 287],
    milkrecordings: [9.8, 26.3, 31.0, 30.0, 30.8, 32.8, 31.4, 31.0, 30.9, 33.6, 33.1],
    source: "icar",
  },
  {
    id: "icar-2212",
    label: "Reference cow 2212 - parity 2",
    description: "Anonymized test-day records: second-lactation cow, pronounced peak near 48 kg.",
    parity: 2,
    breed: "H",
    dim: [5, 33, 61, 89, 117, 145, 173, 201, 229, 257, 299],
    milkrecordings: [31.9, 46.9, 48.5, 43.6, 39.7, 38.0, 34.8, 38.4, 39.9, 36.8, 26.9],
    source: "icar",
  },
  {
    id: "icar-1900",
    label: "Reference cow 1900 - parity 4",
    description:
      "Anonymized test-day records: mature cow, late peak reaching 56 kg on the third test.",
    parity: 4,
    breed: "H",
    dim: [3, 31, 59, 87, 115, 143, 171, 199, 227, 255, 283],
    milkrecordings: [28.7, 29.8, 56.0, 49.2, 47.3, 37.8, 37.6, 36.6, 28.2, 30.3, 31.9],
    source: "icar",
  },
  {
    id: "icar-1863",
    label: "Reference cow 1863 - parity 5",
    description: "Anonymized test-day records: fifth-parity cow with a noisier yield profile.",
    parity: 5,
    breed: "H",
    dim: [17, 45, 73, 101, 129, 157, 185, 213, 241, 269],
    milkrecordings: [31.3, 23.6, 28.8, 39.0, 31.9, 35.0, 30.1, 31.0, 33.7, 29.4],
    source: "icar",
  },
  {
    id: "icar-1483",
    label: "Reference cow 1483 - parity 7",
    description:
      "Anonymized test-day records: aged high-yielder peaking at 53 kg then declining sharply.",
    parity: 7,
    breed: "H",
    dim: [15, 43, 71, 99, 127, 155, 183, 211, 239, 268, 302],
    milkrecordings: [49.1, 53.4, 52.1, 48.3, 37.0, 31.1, 22.7, 24.9, 24.8, 24.4, 21.0],
    source: "icar",
  },
] as const;

export const DEFAULT_LACTATION = EXAMPLE_LACTATIONS[1];
