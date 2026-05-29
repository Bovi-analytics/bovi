import { describe, expect, test } from "bun:test";
import { NextRequest } from "next/server";
import { middleware } from "./middleware";

function request(pathname: string): NextRequest {
  return new NextRequest(new URL(pathname, "https://dashboard.example.test"));
}

describe("middleware", () => {
  test("allows public home and contact routes without an auth marker", () => {
    expect(middleware(request("/")).status).toBe(200);
    expect(middleware(request("/contact")).status).toBe(200);
    expect(middleware(request("/join?invite=test-token")).status).toBe(200);
  });

  test("redirects protected dashboard routes without an auth marker", () => {
    const response = middleware(request("/data-upload"));

    expect(response.status).toBe(307);
    expect(response.headers.get("location")).toBe(
      "https://dashboard.example.test/auth/login?next=%2Fdata-upload"
    );
  });
});
