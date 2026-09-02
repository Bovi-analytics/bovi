export interface AuthOrganization {
  id: number;
  name: string;
  role: "Owner" | "Member" | string;
}

export interface TermsAcceptanceStatus {
  accepted: boolean;
  terms_key: string;
  terms_version: string;
  document_sha256: string;
  document_filename: string;
  document_url: string;
  accepted_at: string | null;
}

export interface AuthUser {
  id: number;
  entra_tenant_id: string;
  entra_oid: string;
  account_type: "entra" | "personal" | string;
  email: string | null;
  name: string | null;
  roles: string[];
  is_admin: boolean;
  organizations: AuthOrganization[];
  terms_acceptance: TermsAcceptanceStatus;
}

export interface AuthContextValue {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  selectedOrganizationId: number | "all" | null;
  setSelectedOrganizationId: (organizationId: number | "all" | null) => void;
  getAccessToken: () => Promise<string>;
  logout: () => Promise<void>;
}
