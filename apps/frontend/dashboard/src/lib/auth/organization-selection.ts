import type { AuthOrganization, AuthUser } from "./types";

export type OrganizationSelection = number | "all" | null;

function isDirectMembership(organization: AuthOrganization): boolean {
  return organization.role !== "Admin";
}

export function requiresExternalOrganizationConfirmation(
  user: AuthUser,
  organizationId: number
): boolean {
  if (!user.is_admin) return false;
  const organization = user.organizations.find((item) => item.id === organizationId);
  return organization?.role === "Admin";
}

export function getInitialOrganizationSelection(
  user: AuthUser,
  saved: string | null
): OrganizationSelection {
  if (saved === "all" && user.is_admin) return "all";

  if (saved) {
    const savedId = Number.parseInt(saved, 10);
    const savedOrganization = user.organizations.find((item) => item.id === savedId);
    if (savedOrganization && !requiresExternalOrganizationConfirmation(user, savedId)) {
      return savedId;
    }
  }

  const directMemberships = user.organizations.filter(isDirectMembership);
  return directMemberships.length === 1 ? directMemberships[0].id : null;
}
