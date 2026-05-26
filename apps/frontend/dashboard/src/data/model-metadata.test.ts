import { describe, expect, test } from "bun:test";
import { LACTATION_CURVE_DOCUMENTATION_URL } from "./model-metadata";

describe("model metadata", () => {
  test("links classical model details to the lactationcurve package documentation", () => {
    const url = new URL(LACTATION_CURVE_DOCUMENTATION_URL);

    expect(url.hostname).toBe("github.com");
    expect(url.pathname).toContain("/Bovi-analytics/bovi/tree/main/packages/models/lactationcurve");
  });
});
