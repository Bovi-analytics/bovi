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
