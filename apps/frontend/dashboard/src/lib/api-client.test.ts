import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/service", () => ({
  getBackendAccessToken: vi.fn(async () => "test-token"),
  handleUnauthorizedResponse: vi.fn(),
}));

import { handleUnauthorizedResponse } from "@/lib/auth/service";
import { downloadChallengeExport, listChallenges } from "./api-client";

describe("api-client authentication", () => {
  beforeEach(() => {
    process.env["NEXT_PUBLIC_API_URL"] = "https://api.example.test";
    vi.restoreAllMocks();
  });

  it("attaches bearer tokens to JSON API calls", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json([
        {
          id: 1,
          dataset: "icar",
          size: "full",
          period: "all",
          name: "Challenge",
          source: "preset",
          user_id: 1,
          organization_id: 1,
          created_at: null,
        },
      ])
    );
    vi.stubGlobal("fetch", fetchMock);

    await listChallenges();

    expect(fetchMock).toHaveBeenCalledWith("https://api.example.test/benchmark/challenges", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer test-token",
      },
    });
  });

  it("redirects through auth handler on 401 responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json({ detail: "Unauthorized" }, { status: 401 }))
    );

    await expect(listChallenges()).rejects.toThrow("API error 401");
    expect(handleUnauthorizedResponse).toHaveBeenCalled();
  });

  it("downloads challenge exports with bearer token and response filename", async () => {
    const click = vi.fn();
    const revokeObjectURL = vi.fn();
    const createObjectURL = vi.fn(() => "blob:download");
    const anchor = { click, download: "", href: "" } as unknown as HTMLAnchorElement;
    const createElement = vi.spyOn(document, "createElement").mockReturnValue(anchor);
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
    const fetchMock = vi.fn(async () =>
      new Response("cow_id\n1", {
        headers: { "Content-Disposition": 'attachment; filename="challenge_7.csv"' },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await downloadChallengeExport(7);

    expect(fetchMock).toHaveBeenCalledWith("https://api.example.test/benchmark/challenges/7/export", {
      headers: { Authorization: "Bearer test-token" },
    });
    expect(anchor.href).toBe("blob:download");
    expect(anchor.download).toBe("challenge_7.csv");
    expect(click).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:download");
    createElement.mockRestore();
  });
});
