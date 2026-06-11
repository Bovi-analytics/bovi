import { describe, expect, test } from "bun:test";
import { DEFAULT_THEME, THEME_OPTIONS, THEME_STORAGE_KEY, getThemeOption } from "./theme";

describe("dashboard theme registry", () => {
  test("defines a stable persisted system-default theme setup", () => {
    expect(THEME_STORAGE_KEY).toBe("bovi-dashboard-color-scheme");
    expect(DEFAULT_THEME).toBe("auto");
    expect(THEME_OPTIONS.map((option) => option.value)).toEqual(["auto", "light", "dark"]);
  });

  test("falls back to the system option for unknown values", () => {
    expect(getThemeOption("light").label).toBe("Light");
    expect(getThemeOption("dark").label).toBe("Dark");
    expect(getThemeOption("auto").label).toBe("System");
  });
});
