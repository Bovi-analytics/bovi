import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

const AUTH_MARKER_COOKIE = "auth_marker";

export function middleware(request: NextRequest): NextResponse {
  const { pathname } = request.nextUrl;
  const authDisabled =
    process.env["AUTH_DISABLED"] === "true" || process.env["NEXT_PUBLIC_AUTH_DISABLED"] === "true";

  if (authDisabled || pathname.startsWith("/auth")) {
    return NextResponse.next();
  }

  const authMarker = request.cookies.get(AUTH_MARKER_COOKIE);
  if (!authMarker?.value) {
    return NextResponse.redirect(new URL("/auth/login", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
