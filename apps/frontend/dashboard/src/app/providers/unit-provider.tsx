"use client";

import { createContext, useContext, useState, useCallback, useEffect } from "react";
import type { ReactElement, ReactNode } from "react";
import type { WeightUnit } from "@/lib/units";

interface UnitContextValue {
  readonly weightUnit: WeightUnit;
  readonly toggleWeightUnit: () => void;
}

const UnitContext = createContext<UnitContextValue>({
  weightUnit: "kg",
  toggleWeightUnit: () => {},
});

const STORAGE_KEY = "bovi-weight-unit";

interface UnitProviderProps {
  readonly children: ReactNode;
}

export function UnitProvider({ children }: UnitProviderProps): ReactElement {
  const [weightUnit, setWeightUnit] = useState<WeightUnit>("kg");

  // Hydrate from localStorage after mount (avoids SSR mismatch)
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "lbs" || stored === "kg") {
      setWeightUnit(stored);
    }
  }, []);

  const toggleWeightUnit = useCallback(() => {
    setWeightUnit((prev) => {
      const next = prev === "kg" ? "lbs" : "kg";
      localStorage.setItem(STORAGE_KEY, next);
      return next;
    });
  }, []);

  return (
    <UnitContext.Provider value={{ weightUnit, toggleWeightUnit }}>
      {children}
    </UnitContext.Provider>
  );
}

export function useWeightUnit(): UnitContextValue {
  return useContext(UnitContext);
}
