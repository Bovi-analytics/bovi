export function getApiBaseUrl(): string {
  const url = process.env["NEXT_PUBLIC_API_URL"];
  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_API_URL is not set. Copy .env.local.example to .env.local and fill it in."
    );
  }
  // Strip trailing slashes so callers can do `${base}/path` safely
  return url.replace(/\/+$/, "");
}
