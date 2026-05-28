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
      ["Admin", "/admin"],
      ["Organization", "/organization"],
      ["Contact", "/contact"],
    ]);
  });

  test("marks admin navigation as admin-only", () => {
    expect(DASHBOARD_NAVIGATION.find((item) => item.href === "/admin")?.adminOnly).toBe(true);
  });
});
