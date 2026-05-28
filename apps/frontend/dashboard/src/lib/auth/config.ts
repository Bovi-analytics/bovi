import type { Configuration, RedirectRequest } from "@azure/msal-browser";
import { LogLevel } from "@azure/msal-browser";

const clientId = process.env["NEXT_PUBLIC_AZURE_AD_CLIENT_ID"] ?? "";
const configuredScope = process.env["NEXT_PUBLIC_AZURE_AD_API_SCOPE"];
const apiScope = configuredScope ?? (clientId ? `api://${clientId}/access_as_user` : "");

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
    redirectUri: typeof window !== "undefined" ? window.location.origin : "",
    postLogoutRedirectUri: typeof window !== "undefined" ? `${window.location.origin}/auth/login` : "",
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
