export function getApiBaseUrl(): string {
  return "/api/bovi";
}

export function getRuntimeApiBaseUrl(): string {
  const url = process.env["NEXT_PUBLIC_API_URL"];
  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_API_URL is not set. Configure it on the dashboard runtime environment."
    );
  }
  return url.replace(/\/+$/, "");
}

export function buildRuntimeApiUrl(path: string[], search = ""): string {
  const normalizedPath = path.map((segment) => encodeURIComponent(segment)).join("/");
  const suffix = normalizedPath ? `/${normalizedPath}` : "";
  return `${getRuntimeApiBaseUrl()}${suffix}${search}`;
}
