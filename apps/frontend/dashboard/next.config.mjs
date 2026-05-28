import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

function loadRootEnvFallbacks() {
  const dashboardDir = dirname(fileURLToPath(import.meta.url));
  const rootEnvPath = resolve(dashboardDir, "../../..", ".env");
  let content = "";
  try {
    content = readFileSync(rootEnvPath, "utf8");
  } catch {
    return;
  }
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
    const [rawKey, ...rawValue] = trimmed.split("=");
    const key = rawKey.trim();
    if (process.env[key] !== undefined) continue;
    process.env[key] = rawValue.join("=").trim().replace(/^["']|["']$/g, "");
  }
}

loadRootEnvFallbacks();

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_AZURE_AD_CLIENT_ID:
      process.env.NEXT_PUBLIC_AZURE_AD_CLIENT_ID ?? process.env.AZURE_AD_CLIENT_ID ?? "",
    NEXT_PUBLIC_AZURE_AD_API_SCOPE: process.env.NEXT_PUBLIC_AZURE_AD_API_SCOPE ?? "",
    NEXT_PUBLIC_AUTH_DISABLED:
      process.env.NEXT_PUBLIC_AUTH_DISABLED ?? process.env.AUTH_DISABLED ?? "false",
    NEXT_PUBLIC_DEV_MODE: process.env.NEXT_PUBLIC_DEV_MODE ?? process.env.DEV_MODE ?? "false",
  },
};

export default nextConfig;
