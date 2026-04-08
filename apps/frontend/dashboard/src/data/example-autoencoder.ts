export interface ExampleAutoencoderData {
  readonly id: string;
  readonly label: string;
  readonly parity: number;
  readonly herdId?: number;
  readonly milk: readonly (number | null)[];
  readonly events?: readonly string[];
}

/**
 * Example datasets for autoencoder prediction.
 * Example 1 sourced from animal_001.json (316 elements, with null gaps).
 * Example 2 is a synthetic simple lactation curve.
 */
export const EXAMPLE_AUTOENCODER_DATA: readonly ExampleAutoencoderData[] = [
  {
    id: "example-1",
    label: "Holstein -- Parity 2 (CX123)",
    parity: 2,
    herdId: 2942694,
    milk: [
      15.0, 26.0, 26.0, 26.0, 31.0, 35.0, 37.0, 37.0, 38.0, 38.0,
      39.0, 39.0, 40.0, 43.0, 41.0, 42.0, 43.0, 43.0, null, null,
      null, null, null, null, null, null, null, null, null, null,
      null, null, null, null, null, null, null, null, null, null,
      null, null, null, null, null, null, null, null, null, null,
      null, null, 50.0, 50.0, 51.0, 49.0, 49.0, 49.0, 48.0, 50.0,
      48.0, 47.0, 47.0, 48.0, 47.0, 48.0, 48.0, 51.0, 49.0, 47.0,
      50.0, 50.0, 49.0, 50.0, 49.0, 48.0, 47.0, 46.0, 45.0, 49.0,
      46.0, 45.0, 47.0, 44.0, 45.0, 47.0, 44.0, 45.0, 47.0, 47.0,
      49.0, 48.0, 47.0, 49.0, 47.0, 46.0, 46.0, 45.0, 46.0, 47.0,
      45.0, 45.0, 46.0, 45.0, 42.0, 43.0, 41.0, 41.0, 44.0, 43.0,
      43.0, 43.0, 41.0, 41.0, 43.0, 43.0, 43.0, 43.0, 42.0, 45.0,
      45.0, 45.0, 45.0, 46.0, 45.0, 46.0, 46.0, 45.0, 45.0, 45.0,
      44.0, 43.0, 44.0, 44.0, 44.0, 45.0, 45.0, 40.0, 41.0, 42.0,
      43.0, 42.0, 42.0, 43.0, 43.0, 43.0, 43.0, 40.0, 40.0, 42.0,
      39.0, 42.0, 41.0, 40.0, 41.0, 39.0, 38.0, 38.0, 37.0, 37.0,
      null, null, null, null, null, null, null, null, null, null,
      null, null, null, null, null, null, null, null, null, null,
      null, null, null, null, null, null, null, null, null, null,
      null, null, null, null, null, null, null, null, null, null,
      null, null, null, null, null, null, null, null, null, null,
      null, null, null, null, null, null, null, null, null, null,
      null, null, null, null, null, null, null, null, null, null,
      null, null, null, null, null, null, null, null, null, null,
      null, null, 36.0, 35.0, 37.0, 35.0, 35.0, 35.0, 36.0, 34.0,
      34.0, 34.0, 34.0, 33.0, 33.0, 32.0, 34.0, 34.0, 35.0, 34.0,
      34.0, 34.0, 35.0, 34.0, 34.0, 34.0, 35.0, 33.0, 33.0, 32.0,
      32.0, 31.0, 32.0, 32.0, 31.0, 31.0, 31.0, 30.0, 31.0, 31.0,
      31.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 30.0, 29.0,
      27.0, 27.0, 26.0, 27.0, 27.0, 26.0, 25.0, 27.0, 26.0, 25.0,
      25.0, 25.0, 24.0, 24.0, 23.0, 22.0, 23.0, 21.0, 19.0, 19.0,
      18.0, 18.0, 18.0, 18.0, 16.0, 15.0,
    ],
    events: [
      "Calving", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "Breeding", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "Breeding", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PregnancyPositive", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "Breeding", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "Breeding", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PregnancyPositive", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
      "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD",
    ],
  },
  {
    id: "example-2",
    label: "Holstein -- Parity 1 (simple curve)",
    parity: 1,
    milk: [
      20.0, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5, 30.0, 31.0, 32.0,
      33.0, 34.0, 34.5, 35.0, 35.5, 36.0, 36.0, 36.0, 35.5, 35.0,
      34.5, 34.0, 33.5, 33.0, 32.5, 32.0, 31.5, 31.0, 30.5, 30.0,
      29.5, 29.0, 28.5, 28.0, 27.5, 27.0, 26.5, 26.0, 25.5, 25.0,
      24.5, 24.0, 23.5, 23.0, 22.5, 22.0, 21.5, 21.0, 20.5, 20.0,
      19.5, 19.0, 18.5, 18.0, 17.5, 17.0, 16.5, 16.0, 15.5, 15.0,
    ],
  },
] as const;

export const DEFAULT_AUTOENCODER_DATA = EXAMPLE_AUTOENCODER_DATA[0];
