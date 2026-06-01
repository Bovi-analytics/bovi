import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/service", () => ({
  getBackendAccessToken: vi.fn(async () => "test-token"),
  handleUnauthorizedResponse: vi.fn(),
}));

import { handleUnauthorizedResponse } from "@/lib/auth/service";
import {
  createOrganizationInvite,
  downloadChallengeExport,
  getInvitePreview,
  listAdminSubmissionsOverview,
  listAdminUsers,
  listChallenges,
  updateAdminUserRole,
  updateOrganizationMemberRole,
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

  it("loads invite preview without bearer auth", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        organization_id: 1,
        organization_name: "Test Organization",
        role: "Member",
        expires_at: "2026-06-01T10:00:00Z",
      })
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await expect(getInvitePreview("invite token")).resolves.toMatchObject({
      organization_name: "Test Organization",
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/bovi/invites/invite%20token/preview", {
      method: "GET",
    });
  });

  it("creates organization invites with the selected role", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        id: 1,
        organization_id: 1,
        created_by_user_id: 1,
        role: "Owner",
        created_at: "2026-05-29T10:00:00Z",
        expires_at: "2026-06-28T10:00:00Z",
        revoked_at: null,
        accepted_count: 0,
        last_accepted_at: null,
        token: "invite-token",
      })
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await expect(createOrganizationInvite(1, "Owner")).resolves.toMatchObject({
      role: "Owner",
      token: "invite-token",
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/bovi/organizations/1/invites", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer test-token",
      },
      body: JSON.stringify({ role: "Owner" }),
    });
  });

  it("updates organization member roles", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        user_id: 2,
        email: "member@example.test",
        name: "Member User",
        role: "Owner",
      })
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await expect(updateOrganizationMemberRole(1, 2, "Owner")).resolves.toMatchObject({
      role: "Owner",
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/bovi/organizations/1/members/2", {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer test-token",
      },
      body: JSON.stringify({ role: "Owner" }),
    });
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

  it("loads admin users with search", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json([
        {
          id: 1,
          entra_tenant_id: "tenant-a",
          entra_oid: "oid-a",
          account_type: "entra",
          email: "admin@example.test",
          name: "Admin User",
          role: "Admin",
          last_login_at: null,
          memberships: [],
        },
      ])
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await expect(listAdminUsers("admin")).resolves.toHaveLength(1);

    expect(fetchMock).toHaveBeenCalledWith("/api/bovi/admin/users?q=admin", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer test-token",
      },
    });
  });

  it("updates global admin user roles", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        id: 1,
        entra_tenant_id: "tenant-a",
        entra_oid: "oid-a",
        account_type: "entra",
        email: "admin@example.test",
        name: "Admin User",
        role: "User",
        last_login_at: null,
        memberships: [],
      })
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await expect(updateAdminUserRole(1, "User")).resolves.toMatchObject({ role: "User" });

    expect(fetchMock).toHaveBeenCalledWith("/api/bovi/admin/users/1/role", {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer test-token",
      },
      body: JSON.stringify({ role: "User" }),
    });
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
