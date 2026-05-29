export function getApiBaseUrl(): string {
  return "/api/bovi";
}

export function getRuntimeApiBaseUrl(): string {
  const url = process.env["API_URL"];
  if (!url) {
    throw new Error("API_URL is not set. Configure it on the dashboard runtime environment.");
  }
  return url.replace(/\/+$/, "");
}

export function buildRuntimeApiUrl(path: string[], search = "", trailingSlash = false): string {
  const normalizedPath = path.map((segment) => encodeURIComponent(segment)).join("/");
  const suffix = normalizedPath ? `/${normalizedPath}${trailingSlash ? "/" : ""}` : "";
  return `${getRuntimeApiBaseUrl()}${suffix}${search}`;
}
