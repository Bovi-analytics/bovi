export interface AuthOrganization {
  id: number;
  name: string;
  role: "Owner" | "Member" | string;
}

export interface AuthUser {
  id: number;
  entra_oid: string;
  email: string | null;
  name: string | null;
  roles: string[];
  is_admin: boolean;
  organizations: AuthOrganization[];
}

export interface AuthContextValue {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  selectedOrganizationId: number | null;
  setSelectedOrganizationId: (organizationId: number) => void;
  getAccessToken: () => Promise<string>;
  logout: () => Promise<void>;
}
