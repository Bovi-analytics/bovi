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
};

export default nextConfig;
