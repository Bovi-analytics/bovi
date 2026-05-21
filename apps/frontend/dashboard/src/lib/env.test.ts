import { afterEach, describe, expect, test } from "bun:test";
import { buildRuntimeApiUrl, getApiBaseUrl, getRuntimeApiBaseUrl } from "./env";

const ORIGINAL_API_URL = process.env["NEXT_PUBLIC_API_URL"];

afterEach(() => {
  if (ORIGINAL_API_URL === undefined) {
    delete process.env["NEXT_PUBLIC_API_URL"];
    return;
  }
  process.env["NEXT_PUBLIC_API_URL"] = ORIGINAL_API_URL;
});

describe("dashboard API environment", () => {
  test("uses the local Next proxy for browser calls", () => {
    expect(getApiBaseUrl()).toBe("/api/bovi");
  });

  test("reads and normalizes the runtime API URL on the server", () => {
    process.env["NEXT_PUBLIC_API_URL"] = "https://api.example.test///";

    expect(getRuntimeApiBaseUrl()).toBe("https://api.example.test");
  });

  test("builds proxied runtime API URLs with path and query string", () => {
    process.env["NEXT_PUBLIC_API_URL"] = "https://api.example.test";

    expect(buildRuntimeApiUrl(["benchmark", "submissions"], "?limit=5")).toBe(
      "https://api.example.test/benchmark/submissions?limit=5"
    );
  });

  test("requires runtime API URL for the server proxy", () => {
    delete process.env["NEXT_PUBLIC_API_URL"];

    expect(() => getRuntimeApiBaseUrl()).toThrow("NEXT_PUBLIC_API_URL is not set");
  });
});
