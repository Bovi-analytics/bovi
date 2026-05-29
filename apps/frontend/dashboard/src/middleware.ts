import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

const AUTH_MARKER_COOKIE = "auth_marker";

export function middleware(request: NextRequest): NextResponse {
  const { pathname } = request.nextUrl;
  const authDisabled = process.env["AUTH_DISABLED"] === "true";

  if (
    authDisabled ||
    pathname === "/" ||
    pathname === "/contact" ||
    pathname === "/join" ||
    pathname.startsWith("/auth") ||
    /\.[^/]+$/.test(pathname)
  ) {
    return NextResponse.next();
  }

  const authMarker = request.cookies.get(AUTH_MARKER_COOKIE);
  if (!authMarker?.value) {
    const loginUrl = new URL("/auth/login", request.url);
    loginUrl.searchParams.set("next", `${pathname}${request.nextUrl.search}`);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
