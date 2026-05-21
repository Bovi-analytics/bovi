import { describe, expect, test } from "bun:test";
import { DASHBOARD_NAVIGATION } from "./navigation";

describe("DASHBOARD_NAVIGATION", () => {
  test("orders data upload before herd profiles and curves", () => {
    expect(DASHBOARD_NAVIGATION.map((item) => [item.label, item.href])).toEqual([
      ["Home", "/"],
      ["Data Upload", "/data-upload"],
      ["Herd Profiles", "/herd-profiles"],
      ["Curves", "/curves"],
      ["Benchmark", "/benchmark"],
    ]);
  });
});
