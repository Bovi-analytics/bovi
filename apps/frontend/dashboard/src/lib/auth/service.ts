import type { IPublicClientApplication } from "@azure/msal-browser";
import { InteractionRequiredAuthError } from "@azure/msal-browser";
import { backendApiRequest } from "./config";

const AUTH_MARKER_COOKIE = "auth_marker";

let msalInstance: IPublicClientApplication | null = null;
let devToken: string | null = null;

export function initializeAuthService(instance: IPublicClientApplication): void {
  msalInstance = instance;
}

export function setDevAccessToken(token: string | null): void {
  devToken = token;
}

export function setAuthMarker(authenticated: boolean): void {
  if (typeof document === "undefined") return;
  if (authenticated) {
    const expires = new Date(Date.now() + 24 * 60 * 60 * 1000);
    document.cookie = `${AUTH_MARKER_COOKIE}=1; expires=${expires.toUTCString()}; path=/; SameSite=Lax`;
    return;
  }
  document.cookie = `${AUTH_MARKER_COOKIE}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`;
}

export async function getBackendAccessToken(): Promise<string> {
  if (devToken) return devToken;
  if (!msalInstance) {
    throw new Error("Authentication service is not initialized.");
  }
  const account = msalInstance.getActiveAccount();
  if (!account) {
    throw new Error("No active account.");
  }

  try {
    const response = await msalInstance.acquireTokenSilent({ ...backendApiRequest, account });
    return response.accessToken;
  } catch (error) {
    if (error instanceof InteractionRequiredAuthError) {
      await msalInstance.acquireTokenRedirect(backendApiRequest);
      throw new Error("Token acquisition requires redirect.");
    }
    throw error;
  }
}

export function handleUnauthorizedResponse(): void {
  if (typeof window === "undefined") return;
  setAuthMarker(false);
  window.location.href = "/auth/login";
}
