export const POST_LOGIN_REDIRECT_KEY = "bovi:post-login-redirect";

export function getSafePostLoginRedirect(value: string | null): string | null {
  if (!value || !value.startsWith("/") || value.startsWith("//") || value.startsWith("/auth")) {
    return null;
  }
  return value;
}
