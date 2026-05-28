"use client";

import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { AuthenticationResult } from "@azure/msal-browser";
import { EventType, PublicClientApplication } from "@azure/msal-browser";
import { MsalProvider, useIsAuthenticated, useMsal } from "@azure/msal-react";
import { getApiBaseUrl } from "@/lib/env";
import { isAuthDisabled, isAzureAdConfigured, loginRequest, msalConfig } from "./config";
import {
  getBackendAccessToken,
  initializeAuthService,
  setAuthMarker,
  setDevAccessToken,
} from "./service";
import type { AuthContextValue, AuthUser } from "./types";

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const DEV_USER: AuthUser = {
  id: 1,
  entra_tenant_id: "dev-tenant",
  entra_oid: "dev-user",
  account_type: "entra",
  email: "dev@local.test",
  name: "Development User",
  roles: ["Admin"],
  is_admin: true,
  organizations: [{ id: 1, name: "Development Organization", role: "Owner" }],
};
const SELECTED_ORG_KEY = "bovi:selected-organization-id";

async function fetchMe(token: string): Promise<AuthUser> {
  const response = await fetch(`${getApiBaseUrl()}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch current user: ${response.status}`);
  }
  return (await response.json()) as AuthUser;
}

function AuthProvider({ children }: { readonly children: ReactNode }) {
  const { instance } = useMsal();
  const isMsalAuthenticated = useIsAuthenticated();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedOrganizationId, setSelectedOrganizationIdState] = useState<number | "all" | null>(
    null
  );

  useEffect(() => {
    initializeAuthService(instance);
  }, [instance]);

  useEffect(() => {
    const loadUser = async () => {
      setIsLoading(true);
      const account = instance.getActiveAccount();
      if (!account) {
        setUser(null);
        setAuthMarker(false);
        setIsLoading(false);
        return;
      }
      try {
        const token = await getBackendAccessToken();
        const currentUser = await fetchMe(token);
        setUser(currentUser);
        const saved = window.localStorage.getItem(SELECTED_ORG_KEY);
        const savedSelection =
          saved === "all" && currentUser.is_admin
            ? "all"
            : saved
              ? Number.parseInt(saved, 10)
              : null;
        const isValidSavedSelection =
          savedSelection === "all" ||
          (typeof savedSelection === "number" &&
            currentUser.organizations.some((org) => org.id === savedSelection));
        setSelectedOrganizationIdState(
          isValidSavedSelection ? savedSelection : (currentUser.organizations[0]?.id ?? null)
        );
        setAuthMarker(true);
      } catch (error) {
        console.error("Failed to load authenticated user", error);
        setUser(null);
        setAuthMarker(false);
      } finally {
        setIsLoading(false);
      }
    };
    void loadUser();
  }, [instance, isMsalAuthenticated]);

  const logout = useCallback(async () => {
    setAuthMarker(false);
    const account = instance.getActiveAccount();
    await instance.logoutRedirect({
      account: account ?? undefined,
      postLogoutRedirectUri: `${window.location.origin}/auth/login`,
    });
  }, [instance]);

  const setSelectedOrganizationId = useCallback((organizationId: number | "all" | null) => {
    if (organizationId === null) {
      window.localStorage.removeItem(SELECTED_ORG_KEY);
    } else {
      window.localStorage.setItem(SELECTED_ORG_KEY, String(organizationId));
    }
    setSelectedOrganizationIdState(organizationId);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isAuthenticated: Boolean(user && isMsalAuthenticated),
      isLoading,
      selectedOrganizationId,
      setSelectedOrganizationId,
      getAccessToken: getBackendAccessToken,
      logout,
    }),
    [
      isLoading,
      isMsalAuthenticated,
      logout,
      selectedOrganizationId,
      setSelectedOrganizationId,
      user,
    ]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

function DevAuthProvider({ children }: { readonly children: ReactNode }) {
  const [selectedOrganizationId, setSelectedOrganizationId] = useState<number | "all" | null>(1);

  useEffect(() => {
    setDevAccessToken("dev-token");
    setAuthMarker(true);
    return () => setDevAccessToken(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user: DEV_USER,
      isAuthenticated: true,
      isLoading: false,
      selectedOrganizationId,
      setSelectedOrganizationId,
      getAccessToken: getBackendAccessToken,
      logout: async () => setAuthMarker(false),
    }),
    [selectedOrganizationId]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function AuthProviderWrapper({ children }: { readonly children: ReactNode }) {
  const [instance, setInstance] = useState<PublicClientApplication | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);

  useEffect(() => {
    if (isAuthDisabled()) {
      setIsInitialized(true);
      return;
    }
    if (!isAzureAdConfigured()) {
      setInitError("Microsoft Entra ID is not configured.");
      setIsInitialized(true);
      return;
    }

    const init = async () => {
      try {
        const runtimeConfig = {
          ...msalConfig,
          auth: {
            ...msalConfig.auth,
            redirectUri: window.location.origin,
            postLogoutRedirectUri: `${window.location.origin}/auth/login`,
          },
        };
        const msal = new PublicClientApplication(runtimeConfig);
        await msal.initialize();
        const response = await msal.handleRedirectPromise();
        if (response) {
          msal.setActiveAccount(response.account);
        }
        const accounts = msal.getAllAccounts();
        if (!msal.getActiveAccount() && accounts[0]) {
          msal.setActiveAccount(accounts[0]);
        }
        msal.addEventCallback((event) => {
          if (event.eventType === EventType.LOGIN_SUCCESS && event.payload) {
            msal.setActiveAccount((event.payload as AuthenticationResult).account);
          }
        });
        setInstance(msal);
      } catch (error) {
        setInitError(error instanceof Error ? error.message : String(error));
      } finally {
        setIsInitialized(true);
      }
    };
    void init();
  }, []);

  if (!isInitialized) return null;
  if (isAuthDisabled()) return <DevAuthProvider>{children}</DevAuthProvider>;
  if (initError || !instance) {
    return <div className="p-6 text-sm text-red-400">Authentication error: {initError}</div>;
  }
  return (
    <MsalProvider instance={instance}>
      <AuthProvider>{children}</AuthProvider>
    </MsalProvider>
  );
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProviderWrapper.");
  return context;
}

export async function startLogin(): Promise<void> {
  if (!msalConfig.auth.clientId) throw new Error("Microsoft Entra ID is not configured.");
  const instance = new PublicClientApplication(msalConfig);
  await instance.initialize();
  await instance.loginRedirect(loginRequest);
}
