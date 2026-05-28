import { afterEach, describe, expect, test } from "bun:test";

const ORIGINAL_REDIRECT_URI = process.env["NEXT_PUBLIC_AZURE_AD_REDIRECT_URI"];
const ORIGINAL_POST_LOGOUT_REDIRECT_URI =
  process.env["NEXT_PUBLIC_AZURE_AD_POST_LOGOUT_REDIRECT_URI"];
const ORIGINAL_CLIENT_ID = process.env["NEXT_PUBLIC_AZURE_AD_CLIENT_ID"];
const ORIGINAL_API_SCOPE = process.env["NEXT_PUBLIC_AZURE_AD_API_SCOPE"];
const ORIGINAL_AUTH_DISABLED = process.env["NEXT_PUBLIC_AUTH_DISABLED"];
const ORIGINAL_DEV_MODE = process.env["NEXT_PUBLIC_DEV_MODE"];

async function loadConfig() {
  const modulePath = `./config?test=${Date.now()}-${Math.random()}`;
  return await import(modulePath);
}

afterEach(() => {
  if (ORIGINAL_REDIRECT_URI === undefined) {
    delete process.env["NEXT_PUBLIC_AZURE_AD_REDIRECT_URI"];
  } else {
    process.env["NEXT_PUBLIC_AZURE_AD_REDIRECT_URI"] = ORIGINAL_REDIRECT_URI;
  }
  if (ORIGINAL_POST_LOGOUT_REDIRECT_URI === undefined) {
    delete process.env["NEXT_PUBLIC_AZURE_AD_POST_LOGOUT_REDIRECT_URI"];
  } else {
    process.env["NEXT_PUBLIC_AZURE_AD_POST_LOGOUT_REDIRECT_URI"] =
      ORIGINAL_POST_LOGOUT_REDIRECT_URI;
  }
  if (ORIGINAL_CLIENT_ID === undefined) {
    delete process.env["NEXT_PUBLIC_AZURE_AD_CLIENT_ID"];
  } else {
    process.env["NEXT_PUBLIC_AZURE_AD_CLIENT_ID"] = ORIGINAL_CLIENT_ID;
  }
  if (ORIGINAL_API_SCOPE === undefined) {
    delete process.env["NEXT_PUBLIC_AZURE_AD_API_SCOPE"];
  } else {
    process.env["NEXT_PUBLIC_AZURE_AD_API_SCOPE"] = ORIGINAL_API_SCOPE;
  }
  if (ORIGINAL_AUTH_DISABLED === undefined) {
    delete process.env["NEXT_PUBLIC_AUTH_DISABLED"];
  } else {
    process.env["NEXT_PUBLIC_AUTH_DISABLED"] = ORIGINAL_AUTH_DISABLED;
  }
  if (ORIGINAL_DEV_MODE === undefined) {
    delete process.env["NEXT_PUBLIC_DEV_MODE"];
  } else {
    process.env["NEXT_PUBLIC_DEV_MODE"] = ORIGINAL_DEV_MODE;
  }
});

describe("auth redirect configuration", () => {
  test("defaults the auth redirect to the login route on the current origin", async () => {
    delete process.env["NEXT_PUBLIC_AZURE_AD_REDIRECT_URI"];

    const { getAuthRedirectUri } = await loadConfig();

    expect(getAuthRedirectUri("http://localhost:3000")).toBe("http://localhost:3000/auth/login");
  });

  test("resolves relative configured redirect URIs against the current origin", async () => {
    process.env["NEXT_PUBLIC_AZURE_AD_REDIRECT_URI"] = "/auth/callback";

    const { getAuthRedirectUri } = await loadConfig();

    expect(getAuthRedirectUri("https://dashboard.example.test")).toBe(
      "https://dashboard.example.test/auth/callback"
    );
  });

  test("preserves absolute configured redirect URIs", async () => {
    process.env["NEXT_PUBLIC_AZURE_AD_REDIRECT_URI"] =
      "https://dashboard.example.test/auth/login";

    const { getAuthRedirectUri } = await loadConfig();

    expect(getAuthRedirectUri("http://localhost:3000")).toBe(
      "https://dashboard.example.test/auth/login"
    );
  });

  test("defaults the post-logout redirect to the application root", async () => {
    delete process.env["NEXT_PUBLIC_AZURE_AD_POST_LOGOUT_REDIRECT_URI"];

    const { getPostLogoutRedirectUri } = await loadConfig();

    expect(getPostLogoutRedirectUri("http://localhost:3000")).toBe("http://localhost:3000/");
  });

  test("uses runtime auth flags without removing Entra configuration", async () => {
    process.env["NEXT_PUBLIC_AZURE_AD_CLIENT_ID"] = "client-id";
    process.env["NEXT_PUBLIC_AUTH_DISABLED"] = "true";

    const { getDefaultAuthRuntimeConfig, createMsalConfig, isAuthDisabled, isAzureAdConfigured } =
      await loadConfig();

    const runtimeConfig = getDefaultAuthRuntimeConfig();
    expect(isAuthDisabled(runtimeConfig)).toBe(true);
    expect(isAzureAdConfigured(runtimeConfig)).toBe(true);
    expect(createMsalConfig(runtimeConfig).auth.clientId).toBe("client-id");
  });
});
