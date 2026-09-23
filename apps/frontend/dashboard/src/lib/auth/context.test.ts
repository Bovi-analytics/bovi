import { describe, expect, test } from "bun:test";
import type { AuthUser } from "./types";
import {
  getInitialOrganizationSelection,
  requiresExternalOrganizationConfirmation,
} from "./organization-selection";
import { getInitialPostLoginRedirect, getSafePostLoginRedirect } from "./post-login-redirect";

const USER: AuthUser = {
  id: 1,
  entra_tenant_id: "tenant",
  entra_oid: "oid",
  account_type: "entra",
  email: "user@example.test",
  name: "Test User",
  roles: ["User"],
  is_admin: false,
  organizations: [],
  terms_acceptance: {
    accepted: true,
    terms_key: "terms",
    terms_version: "1",
    document_sha256: "hash",
    document_filename: "terms.docx",
    document_url: "/terms.docx",
    accepted_at: null,
  },
};

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

describe("getInitialOrganizationSelection", () => {
  test("selects the only direct membership instead of an admin-only organization", () => {
    const admin = {
      ...USER,
      is_admin: true,
      roles: ["Admin"],
      organizations: [
        { id: 12, name: "Albart Coster", role: "Admin" },
        { id: 2, name: "Corrie cornell", role: "Owner" },
      ],
    };

    expect(getInitialOrganizationSelection(admin, null)).toBe(2);
    expect(getInitialOrganizationSelection(admin, "12")).toBe(2);
    expect(requiresExternalOrganizationConfirmation(admin, 12)).toBe(true);
    expect(requiresExternalOrganizationConfirmation(admin, 2)).toBe(false);
  });

  test("requires an explicit choice when a non-admin has multiple memberships", () => {
    const member = {
      ...USER,
      organizations: [
        { id: 2, name: "North Herd", role: "Owner" },
        { id: 3, name: "South Herd", role: "Member" },
      ],
    };

    expect(getInitialOrganizationSelection(member, null)).toBeNull();
    expect(getInitialOrganizationSelection(member, "3")).toBe(3);
  });

  test("selects a sole membership and preserves the admin all-organizations view", () => {
    expect(
      getInitialOrganizationSelection(
        { ...USER, organizations: [{ id: 2, name: "Only Herd", role: "Member" }] },
        null
      )
    ).toBe(2);
    expect(
      getInitialOrganizationSelection({ ...USER, is_admin: true, roles: ["Admin"] }, "all")
    ).toBe("all");
  });
});
