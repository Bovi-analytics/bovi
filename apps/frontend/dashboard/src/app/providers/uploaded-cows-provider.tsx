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
  readonly id: string;
  readonly name: string;
  readonly format: "icar_test_day";
  readonly uploadedAt: string;
  readonly userId?: number | null;
  readonly userName?: string | null;
  readonly userEmail?: string | null;
  readonly organizationId?: number | null;
  readonly organizationName?: string | null;
  readonly rowCount?: number;
  readonly cowCount?: number;
  readonly detectedParity?: number | null;
  readonly columns?: readonly string[];
  readonly columnMapping?: Readonly<Record<string, string>>;
  readonly stats?: Readonly<Record<string, number>>;
  readonly rawStats?: Readonly<Record<string, number>>;
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
  readonly savedDatasets: readonly UploadedDataset[];
  readonly saveDataset: (d: UploadedDataset) => void;
  readonly selectSavedDataset: (id: string) => void;
  readonly deleteSavedDataset: (id: string) => void;
  readonly getCow: (cowId: string) => UploadedCow | undefined;
  readonly getRandomCow: () => UploadedCow | undefined;
  readonly activePreset: ActivePreset | null;
  readonly setActivePreset: (p: ActivePreset | null) => void;
}

const UploadedCowsContext = createContext<UploadedCowsContextValue>({
  dataset: null,
  setDataset: () => {},
  clearDataset: () => {},
  savedDatasets: [],
  saveDataset: () => {},
  selectSavedDataset: () => {},
  deleteSavedDataset: () => {},
  getCow: () => undefined,
  getRandomCow: () => undefined,
  activePreset: null,
  setActivePreset: () => {},
});

const STORAGE_KEY = "bovi-uploaded-cows-v1";
const SAVED_DATASETS_STORAGE_KEY = "bovi-saved-uploaded-datasets-v1";
const PRESET_STORAGE_KEY = "bovi-active-preset-v1";

interface UploadedCowsProviderProps {
  readonly children: ReactNode;
}

export function UploadedCowsProvider({ children }: UploadedCowsProviderProps): ReactElement {
  const [dataset, setDatasetState] = useState<UploadedDataset | null>(null);
  const [savedDatasets, setSavedDatasetsState] = useState<UploadedDataset[]>([]);
  const [activePreset, setActivePresetState] = useState<ActivePreset | null>(null);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (!stored) return;
      const parsed = JSON.parse(stored) as UploadedDataset;
      if (parsed && Array.isArray(parsed.cows)) {
        setDatasetState({
          ...parsed,
          id: parsed.id || `${parsed.uploadedAt}-${parsed.name}`,
        });
      }
    } catch {
      // Corrupt state - ignore and start fresh.
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(SAVED_DATASETS_STORAGE_KEY);
      if (!stored) return;
      const parsed = JSON.parse(stored) as UploadedDataset[];
      if (Array.isArray(parsed)) {
        setSavedDatasetsState(
          parsed
            .filter((item) => item && Array.isArray(item.cows))
            .map((item) => ({
              ...item,
              id: item.id || `${item.uploadedAt}-${item.name}`,
            }))
        );
      }
    } catch {
      localStorage.removeItem(SAVED_DATASETS_STORAGE_KEY);
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

  const persistSavedDatasets = useCallback((items: UploadedDataset[]) => {
    try {
      localStorage.setItem(SAVED_DATASETS_STORAGE_KEY, JSON.stringify(items));
    } catch {
      // Quota or other storage error - keep the datasets in memory anyway.
    }
  }, []);

  const setDataset = useCallback((d: UploadedDataset) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(d));
      localStorage.removeItem(PRESET_STORAGE_KEY);
    } catch {
      // Quota or other storage error - keep the dataset in memory anyway.
    }
    setDatasetState(d);
    setActivePresetState(null);
  }, []);

  const saveDataset = useCallback(
    (d: UploadedDataset) => {
      setDataset(d);
      setSavedDatasetsState((current) => {
        const next = [d, ...current.filter((item) => item.id !== d.id)].slice(0, 10);
        persistSavedDatasets(next);
        return next;
      });
    },
    [persistSavedDatasets, setDataset]
  );

  const selectSavedDataset = useCallback(
    (id: string) => {
      const saved = savedDatasets.find((item) => item.id === id);
      if (saved) {
        setDataset(saved);
      }
    },
    [savedDatasets, setDataset]
  );

  const clearDataset = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(PRESET_STORAGE_KEY);
    setDatasetState(null);
    setActivePresetState(null);
  }, []);

  const deleteSavedDataset = useCallback(
    (id: string) => {
      setSavedDatasetsState((current) => {
        const next = current.filter((item) => item.id !== id);
        persistSavedDatasets(next);
        return next;
      });
      if (dataset?.id === id) {
        clearDataset();
      }
    },
    [clearDataset, dataset?.id, persistSavedDatasets]
  );

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
      localStorage.removeItem(STORAGE_KEY);
      setDatasetState(null);
    } else {
      localStorage.removeItem(PRESET_STORAGE_KEY);
    }
    setActivePresetState(p);
  }, []);

  return (
    <UploadedCowsContext.Provider
      value={{
        dataset,
        setDataset,
        clearDataset,
        savedDatasets,
        saveDataset,
        selectSavedDataset,
        deleteSavedDataset,
        getCow,
        getRandomCow,
        activePreset,
        setActivePreset,
      }}
    >
      {children}
    </UploadedCowsContext.Provider>
  );
}

export function useUploadedCows(): UploadedCowsContextValue {
  return useContext(UploadedCowsContext);
}
