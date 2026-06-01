import { describe, expect, test } from "bun:test";
import { formatCharacteristicValue, isWeightCharacteristic } from "./characteristics";

describe("characteristic formatting", () => {
  test("formats persistency with three decimal places", () => {
    expect(formatCharacteristicValue("persistency", -0.04362, "kg")).toBe("-0.044");
  });

  test("keeps non-weight non-persistency values at one decimal place", () => {
    expect(formatCharacteristicValue("time_to_peak", 72.34, "kg")).toBe("72.3");
  });

  test("uses weight formatting for yield characteristics", () => {
    expect(isWeightCharacteristic("peak_yield")).toBe(true);
    expect(formatCharacteristicValue("peak_yield", 12.34, "kg")).toBe("12.3");
  });
});
