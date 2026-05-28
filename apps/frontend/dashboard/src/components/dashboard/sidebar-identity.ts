import type { AuthOrganization, AuthUser } from "@/lib/auth/types";

export function getUserDisplayName(user: AuthUser): string {
  return user.name?.trim() || user.email?.trim() || "Signed-in user";
}

export function getUserInitials(user: AuthUser): string {
  const source = user.name?.trim() || user.email?.split("@")[0]?.trim() || "User";
  const parts = source.split(/[\s._-]+/).filter(Boolean);
  const initials = parts.length > 1 ? `${parts[0]?.[0] ?? ""}${parts[1]?.[0] ?? ""}` : source[0];
  return initials?.toUpperCase() ?? "U";
}

export function getSelectedOrganizationLabel(
  selectedOrganizationId: number | "all" | null,
  organizations: readonly AuthOrganization[]
): string {
  if (selectedOrganizationId === "all") return "All organizations";
  if (selectedOrganizationId === null) return "No organization selected";
  return (
    organizations.find((org) => org.id === selectedOrganizationId)?.name ?? "Unknown organization"
  );
}
