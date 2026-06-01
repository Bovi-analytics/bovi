import { formatWeight, type WeightUnit } from "./units";

const WEIGHT_CHARACTERISTICS = new Set(["peak_yield", "cumulative_milk_yield"]);

export function isWeightCharacteristic(name: string): boolean {
  return WEIGHT_CHARACTERISTICS.has(name);
}

export function formatCharacteristicValue(
  name: string,
  value: number,
  weightUnit: WeightUnit
): string {
  if (isWeightCharacteristic(name)) {
    return formatWeight(value, weightUnit);
  }

  const decimals = name === "persistency" ? 3 : 1;
  return value.toFixed(decimals);
}
