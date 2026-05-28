import { describe, expect, test } from "bun:test";
import { getSafePostLoginRedirect } from "./post-login-redirect";

describe("getSafePostLoginRedirect", () => {
  test("allows protected application paths", () => {
    expect(getSafePostLoginRedirect("/curves")).toBe("/curves");
    expect(getSafePostLoginRedirect("/benchmark?scope=mine")).toBe("/benchmark?scope=mine");
  });

  test("rejects missing, external, and auth paths", () => {
    expect(getSafePostLoginRedirect(null)).toBeNull();
    expect(getSafePostLoginRedirect("https://example.test/curves")).toBeNull();
    expect(getSafePostLoginRedirect("//example.test/curves")).toBeNull();
    expect(getSafePostLoginRedirect("/auth/login")).toBeNull();
  });
});
