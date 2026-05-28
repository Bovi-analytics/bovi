import type { Configuration, RedirectRequest } from "@azure/msal-browser";
import { LogLevel } from "@azure/msal-browser";

export interface AuthRuntimeConfig {
  readonly clientId: string;
  readonly apiScope: string;
  readonly authDisabled: boolean;
  readonly redirectUri: string;
  readonly postLogoutRedirectUri: string;
}

function readBoolean(value: string | undefined): boolean {
  return value === "true";
}

export function getDefaultAuthRuntimeConfig(): AuthRuntimeConfig {
  const clientId =
    process.env["NEXT_PUBLIC_AZURE_AD_CLIENT_ID"] ?? process.env["AZURE_AD_CLIENT_ID"] ?? "";
  const configuredScope = process.env["NEXT_PUBLIC_AZURE_AD_API_SCOPE"];
  return {
    clientId,
    apiScope: configuredScope ?? (clientId ? `api://${clientId}/access_as_user` : ""),
    authDisabled:
      readBoolean(process.env["NEXT_PUBLIC_AUTH_DISABLED"]) ||
      readBoolean(process.env["AUTH_DISABLED"]) ||
      readBoolean(process.env["NEXT_PUBLIC_DEV_MODE"]) ||
      readBoolean(process.env["DEV_MODE"]),
    redirectUri: process.env["NEXT_PUBLIC_AZURE_AD_REDIRECT_URI"] ?? "/auth/login",
    postLogoutRedirectUri: process.env["NEXT_PUBLIC_AZURE_AD_POST_LOGOUT_REDIRECT_URI"] ?? "/",
  };
}

let runtimeConfig = getDefaultAuthRuntimeConfig();

export function setAuthRuntimeConfig(config: AuthRuntimeConfig): void {
  runtimeConfig = config;
}

export function getAuthRuntimeConfig(): AuthRuntimeConfig {
  return runtimeConfig;
}

function resolveBrowserUri(value: string, origin?: string): string {
  const browserOrigin = origin ?? (typeof window !== "undefined" ? window.location.origin : "");
  if (!browserOrigin) return "";
  return new URL(value, browserOrigin).toString();
}

export function getAuthRedirectUri(
  origin?: string,
  config: AuthRuntimeConfig = getAuthRuntimeConfig()
): string {
  return resolveBrowserUri(config.redirectUri, origin);
}

export function getPostLogoutRedirectUri(
  origin?: string,
  config: AuthRuntimeConfig = getAuthRuntimeConfig()
): string {
  return resolveBrowserUri(config.postLogoutRedirectUri, origin);
}

export function isAuthDisabled(config: AuthRuntimeConfig = getAuthRuntimeConfig()): boolean {
  return config.authDisabled;
}

export function isAzureAdConfigured(config: AuthRuntimeConfig = getAuthRuntimeConfig()): boolean {
  return Boolean(config.clientId && config.apiScope);
}

export function createMsalConfig(config: AuthRuntimeConfig = getAuthRuntimeConfig()): Configuration {
  return {
    auth: {
      clientId: config.clientId,
      authority: "https://login.microsoftonline.com/common/v2.0",
      redirectUri: getAuthRedirectUri(undefined, config),
      postLogoutRedirectUri: getPostLogoutRedirectUri(undefined, config),
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
}

export function createLoginRequest(
  config: AuthRuntimeConfig = getAuthRuntimeConfig()
): RedirectRequest {
  return {
    scopes: [
      "openid",
      "profile",
      "email",
      "offline_access",
      ...(config.apiScope ? [config.apiScope] : []),
    ],
  };
}

export function createBackendApiRequest(
  config: AuthRuntimeConfig = getAuthRuntimeConfig()
): RedirectRequest {
  return {
    scopes: config.apiScope ? [config.apiScope] : [],
  };
}
