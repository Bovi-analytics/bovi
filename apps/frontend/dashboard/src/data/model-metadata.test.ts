import { describe, expect, test } from "bun:test";
import { MODEL_METADATA } from "./model-metadata";

describe("MODEL_METADATA", () => {
  test("shows the 305-day Ali and Schaeffer formula", () => {
    expect(MODEL_METADATA.ali_schaeffer.formula).toContain("t/305");
    expect(MODEL_METADATA.ali_schaeffer.formula).toContain("ln(305/t)");
    expect(MODEL_METADATA.ali_schaeffer.formula).not.toContain("340");
  });

  test("only MilkBot exposes directly interpreted parameter descriptions", () => {
    expect(MODEL_METADATA.milkbot.parameters.length).toBeGreaterThan(0);

    expect(MODEL_METADATA.wood.parameters).toHaveLength(0);
    expect(MODEL_METADATA.wilmink.parameters).toHaveLength(0);
    expect(MODEL_METADATA.ali_schaeffer.parameters).toHaveLength(0);
    expect(MODEL_METADATA.fischer.parameters).toHaveLength(0);
  });
});
