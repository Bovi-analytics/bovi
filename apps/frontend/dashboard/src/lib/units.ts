const KG_TO_LBS = 2.20462;

export type WeightUnit = "kg" | "lbs";

/**
 * Convert a weight value based on the selected unit.
 * If the unit is "kg", the value is returned as-is.
 */
export function convertWeight(value: number, unit: WeightUnit): number {
  return unit === "lbs" ? value * KG_TO_LBS : value;
}

/**
 * Format a number for display.
 * - "lbs" uses US locale (comma thousands, period decimal).
 * - "kg" uses default locale (period decimal, no thousands separator for small numbers).
 */
export function formatWeight(value: number, unit: WeightUnit, decimals = 1): string {
  const converted = convertWeight(value, unit);
  if (unit === "lbs") {
    return converted.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }
  return converted.toFixed(decimals);
}

/**
 * Get the display label for a weight-based unit string.
 * Replaces "kg" with "lbs" when appropriate.
 */
export function getUnitLabel(baseUnit: string, weightUnit: WeightUnit): string {
  if (weightUnit === "lbs") {
    return baseUnit.replace(/kg/g, "lbs");
  }
  return baseUnit;
}
