import type { Configuration, RedirectRequest } from "@azure/msal-browser";
import { LogLevel } from "@azure/msal-browser";

const clientId = process.env["NEXT_PUBLIC_AZURE_AD_CLIENT_ID"] ?? "";
const configuredScope = process.env["NEXT_PUBLIC_AZURE_AD_API_SCOPE"];
const apiScope = configuredScope ?? (clientId ? `api://${clientId}/access_as_user` : "");
const configuredRedirectUri = process.env["NEXT_PUBLIC_AZURE_AD_REDIRECT_URI"] ?? "/auth/login";
const configuredPostLogoutRedirectUri =
  process.env["NEXT_PUBLIC_AZURE_AD_POST_LOGOUT_REDIRECT_URI"] ?? "/";

function resolveBrowserUri(value: string, origin?: string): string {
  const browserOrigin = origin ?? (typeof window !== "undefined" ? window.location.origin : "");
  if (!browserOrigin) return "";
  return new URL(value, browserOrigin).toString();
}

export function getAuthRedirectUri(origin?: string): string {
  return resolveBrowserUri(configuredRedirectUri, origin);
}

export function getPostLogoutRedirectUri(origin?: string): string {
  return resolveBrowserUri(configuredPostLogoutRedirectUri, origin);
}

export function isAuthDisabled(): boolean {
  return (
    process.env["NEXT_PUBLIC_AUTH_DISABLED"] === "true" ||
    process.env["NEXT_PUBLIC_DEV_MODE"] === "true"
  );
}

export function isAzureAdConfigured(): boolean {
  return Boolean(clientId && apiScope);
}

export const msalConfig: Configuration = {
  auth: {
    clientId,
    authority: "https://login.microsoftonline.com/common/v2.0",
    redirectUri: getAuthRedirectUri(),
    postLogoutRedirectUri: getPostLogoutRedirectUri(),
    navigateToLoginRequestUrl: false,
  },
  cache: {
    cacheLocation: "sessionStorage",
    storeAuthStateInCookie: false,
  },
  system: {
    loggerOptions: {
      loggerCallback: (level, message, containsPii) => {
        if (containsPii || process.env.NODE_ENV === "production") return;
        if (level === LogLevel.Error) console.error("[MSAL]", message);
        if (level === LogLevel.Warning) console.warn("[MSAL]", message);
      },
      logLevel: LogLevel.Warning,
    },
  },
};

export const loginRequest: RedirectRequest = {
  scopes: ["openid", "profile", "email", "offline_access", ...(apiScope ? [apiScope] : [])],
};

export const backendApiRequest: RedirectRequest = {
  scopes: apiScope ? [apiScope] : [],
};
