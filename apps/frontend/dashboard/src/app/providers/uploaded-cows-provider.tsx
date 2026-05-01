"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactElement, ReactNode } from "react";
import type { PresetDatasetKey, PresetPeriodKey, PresetSizeKey } from "@/types/api";

export interface UploadedCow {
  readonly cowId: string;
  readonly parity: number | null;
  readonly dim: readonly number[];
  readonly milkrecordings: readonly number[];
}

export interface UploadedDataset {
  readonly name: string;
  readonly format: "icar_test_day" | "dairycom_test_day";
  readonly uploadedAt: string;
  readonly cows: readonly UploadedCow[];
}

export interface ActivePreset {
  readonly dataset: PresetDatasetKey;
  readonly size: PresetSizeKey;
  readonly period: PresetPeriodKey;
}

interface UploadedCowsContextValue {
  readonly dataset: UploadedDataset | null;
  readonly setDataset: (d: UploadedDataset) => void;
  readonly clearDataset: () => void;
  readonly getCow: (cowId: string) => UploadedCow | undefined;
  readonly getRandomCow: () => UploadedCow | undefined;
  readonly activePreset: ActivePreset | null;
  readonly setActivePreset: (p: ActivePreset | null) => void;
}

const UploadedCowsContext = createContext<UploadedCowsContextValue>({
  dataset: null,
  setDataset: () => {},
  clearDataset: () => {},
  getCow: () => undefined,
  getRandomCow: () => undefined,
  activePreset: null,
  setActivePreset: () => {},
});

const STORAGE_KEY = "bovi-uploaded-cows-v1";
const PRESET_STORAGE_KEY = "bovi-active-preset-v1";

interface UploadedCowsProviderProps {
  readonly children: ReactNode;
}

export function UploadedCowsProvider({ children }: UploadedCowsProviderProps): ReactElement {
  const [dataset, setDatasetState] = useState<UploadedDataset | null>(null);
  const [activePreset, setActivePresetState] = useState<ActivePreset | null>(null);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (!stored) return;
      const parsed = JSON.parse(stored) as UploadedDataset;
      if (parsed && Array.isArray(parsed.cows)) {
        setDatasetState(parsed);
      }
    } catch {
      // Corrupt state — ignore and start fresh.
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(PRESET_STORAGE_KEY);
      if (!stored) return;
      const parsed = JSON.parse(stored) as ActivePreset;
      if (parsed && parsed.dataset && parsed.size && parsed.period) {
        setActivePresetState(parsed);
      }
    } catch {
      localStorage.removeItem(PRESET_STORAGE_KEY);
    }
  }, []);

  const setDataset = useCallback((d: UploadedDataset) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(d));
    } catch {
      // Quota or other storage error — keep the dataset in memory anyway.
    }
    setDatasetState(d);
  }, []);

  const clearDataset = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(PRESET_STORAGE_KEY);
    setDatasetState(null);
    setActivePresetState(null);
  }, []);

  const getCow = useCallback(
    (cowId: string) => dataset?.cows.find((c) => c.cowId === cowId),
    [dataset]
  );

  const getRandomCow = useCallback(() => {
    if (!dataset || dataset.cows.length === 0) return undefined;
    const idx = Math.floor(Math.random() * dataset.cows.length);
    return dataset.cows[idx];
  }, [dataset]);

  const setActivePreset = useCallback((p: ActivePreset | null) => {
    if (p) {
      localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(p));
    } else {
      localStorage.removeItem(PRESET_STORAGE_KEY);
    }
    setActivePresetState(p);
  }, []);

  return (
    <UploadedCowsContext.Provider
      value={{ dataset, setDataset, clearDataset, getCow, getRandomCow, activePreset, setActivePreset }}
    >
      {children}
    </UploadedCowsContext.Provider>
  );
}

export function useUploadedCows(): UploadedCowsContextValue {
  return useContext(UploadedCowsContext);
}
