import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/service", () => ({
  getBackendAccessToken: vi.fn(async () => "test-token"),
  handleUnauthorizedResponse: vi.fn(),
}));

import { handleUnauthorizedResponse } from "@/lib/auth/service";
import {
  downloadChallengeExport,
  listAdminSubmissionsOverview,
  listChallenges,
} from "./api-client";

const ORIGINAL_DOCUMENT = globalThis.document;
const ORIGINAL_FETCH = globalThis.fetch;
const ORIGINAL_URL = globalThis.URL;

describe("api-client authentication", () => {
  beforeEach(() => {
    process.env["API_URL"] = "https://api.example.test";
    vi.restoreAllMocks();
    globalThis.URL = ORIGINAL_URL;
    globalThis.document = ORIGINAL_DOCUMENT;
    if (ORIGINAL_FETCH === undefined) {
      delete (globalThis as { fetch?: typeof fetch }).fetch;
    } else {
      globalThis.fetch = ORIGINAL_FETCH;
    }
  });

  afterEach(() => {
    globalThis.URL = ORIGINAL_URL;
    globalThis.document = ORIGINAL_DOCUMENT;
    if (ORIGINAL_FETCH === undefined) {
      delete (globalThis as { fetch?: typeof fetch }).fetch;
    } else {
      globalThis.fetch = ORIGINAL_FETCH;
    }
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
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await listChallenges(1);

    expect(fetchMock).toHaveBeenCalledWith("/api/bovi/benchmark/challenges?organization_id=1", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer test-token",
      },
    });
  });

  it("redirects through auth handler on 401 responses", async () => {
    globalThis.fetch = vi.fn(async () =>
      Response.json({ detail: "Unauthorized" }, { status: 401 })
    ) as unknown as typeof fetch;

    await expect(listChallenges(1)).rejects.toThrow("API error 401");
    expect(handleUnauthorizedResponse).toHaveBeenCalled();
  });

  it("sends organization list filters to challenge endpoints", async () => {
    const fetchMock = vi.fn(async () => Response.json([]));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await listChallenges(1, {
      scope: "mine",
      sort: "name",
      direction: "asc",
      q: "aurora",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/bovi/benchmark/challenges?organization_id=1&scope=mine&sort=name&direction=asc&q=aurora",
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer test-token",
        },
      }
    );
  });

  it("sends admin overview filters to the admin endpoint", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        kpis: {
          total_items: 0,
          organizations: 0,
          users: 0,
          benchmark_submissions: 0,
          benchmark_challenges: 0,
          herd_dataset_uploads: 0,
          herd_profiles: 0,
          failed_items: 0,
          latest_activity_at: null,
        },
        by_organization: [],
        by_category: [],
        items: [],
      })
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await listAdminSubmissionsOverview({
      organizationId: "all",
      category: "benchmark_submission",
      q: "farm",
      from: "2026-05-01",
      to: "2026-05-28",
      sort: "organization",
      direction: "asc",
      limit: 50,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/bovi/admin/submissions-overview?organization_id=all&category=benchmark_submission&q=farm&from=2026-05-01&to=2026-05-28&sort=organization&direction=asc&limit=50",
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer test-token",
        },
      }
    );
  });

  it("downloads challenge exports with bearer token and response filename", async () => {
    const click = vi.fn();
    const revokeObjectURL = vi.fn();
    const createObjectURL = vi.fn(() => "blob:download");
    const anchor = { click, download: "", href: "" } as unknown as HTMLAnchorElement;
    const documentMock = { createElement: vi.fn(() => anchor) };
    globalThis.document = documentMock as unknown as Document;
    globalThis.URL = { createObjectURL, revokeObjectURL } as unknown as typeof URL;
    const fetchMock = vi.fn(
      async () =>
        new Response("cow_id\n1", {
          headers: { "Content-Disposition": 'attachment; filename="challenge_7.csv"' },
        })
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await downloadChallengeExport(7);

    expect(fetchMock).toHaveBeenCalledWith("/api/bovi/benchmark/challenges/7/export", {
      headers: { Authorization: "Bearer test-token" },
    });
    expect(anchor.href).toBe("blob:download");
    expect(anchor.download).toBe("challenge_7.csv");
    expect(click).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:download");
  });
});
