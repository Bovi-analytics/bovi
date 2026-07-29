import { describe, expect, test } from "bun:test";
import {
  getSelectedOrganizationLabel,
  getUserDisplayName,
  getUserInitials,
} from "./sidebar-identity";
import type { AuthUser } from "@/lib/auth/types";

const USER: AuthUser = {
  id: 1,
  entra_tenant_id: "tenant",
  entra_oid: "oid",
  account_type: "entra",
  email: "jane.doe@example.test",
  name: "Jane Doe",
  roles: ["Member"],
  is_admin: false,
  organizations: [
    { id: 10, name: "North Herd", role: "Owner" },
    { id: 11, name: "South Herd", role: "Member" },
  ],
  terms_acceptance: {
    accepted: true,
    terms_key: "terms-of-use-data-contribution",
    terms_version: "072326",
    document_sha256: "dba8cbba07f6a413d868bfccc4b671f974b48335cc1b5ca2677a73e1ce758304", // pragma: allowlist secret
    document_filename: "Terms of Use and Data Contribution Agreement 072326.docx",
    document_url: "/legal/terms-of-use-data-contribution-agreement-072326.docx",
    accepted_at: "2026-07-29T12:00:00Z",
  },
};

describe("dashboard sidebar identity helpers", () => {
  test("formats signed-in user display labels", () => {
    expect(getUserDisplayName(USER)).toBe("Jane Doe");
    expect(getUserInitials(USER)).toBe("JD");
  });

  test("falls back to email identity when the user has no name", () => {
    expect(getUserDisplayName({ ...USER, name: null })).toBe("jane.doe@example.test");
    expect(getUserInitials({ ...USER, name: null })).toBe("JD");
  });

  test("describes selected organization context", () => {
    expect(getSelectedOrganizationLabel(10, USER.organizations)).toBe("North Herd");
    expect(getSelectedOrganizationLabel("all", USER.organizations)).toBe("All organizations");
    expect(getSelectedOrganizationLabel(null, USER.organizations)).toBe("No organization selected");
  });
});
