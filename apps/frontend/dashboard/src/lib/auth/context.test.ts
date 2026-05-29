import { describe, expect, test } from "bun:test";
import { getInitialPostLoginRedirect, getSafePostLoginRedirect } from "./post-login-redirect";

describe("getSafePostLoginRedirect", () => {
  test("allows protected application paths", () => {
    expect(getSafePostLoginRedirect("/curves")).toBe("/curves");
    expect(getSafePostLoginRedirect("/join?invite=abc123")).toBe("/join?invite=abc123");
    expect(getSafePostLoginRedirect("/benchmark?scope=mine")).toBe("/benchmark?scope=mine");
  });

  test("rejects missing, external, and auth paths", () => {
    expect(getSafePostLoginRedirect(null)).toBeNull();
    expect(getSafePostLoginRedirect("https://example.test/curves")).toBeNull();
    expect(getSafePostLoginRedirect("//example.test/curves")).toBeNull();
    expect(getSafePostLoginRedirect("/auth/login")).toBeNull();
  });
});

describe("getInitialPostLoginRedirect", () => {
  test("prefers query redirects over stored redirects", () => {
    expect(getInitialPostLoginRedirect("?next=%2Fjoin%3Finvite%3Dabc123", "/curves")).toBe(
      "/join?invite=abc123"
    );
  });

  test("uses the stored redirect when the login callback has no next query", () => {
    expect(getInitialPostLoginRedirect("", "/join?invite=abc123")).toBe("/join?invite=abc123");
  });
});
