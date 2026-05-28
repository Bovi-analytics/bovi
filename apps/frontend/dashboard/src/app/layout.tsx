import type { ReactElement, ReactNode } from "react";
import "./globals.css";
import { Providers } from "./providers/root-provider";
import type { AuthRuntimeConfig } from "@/lib/auth/config";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Lactation Curves Dashboard",
  description: "Interactive visualization of lactation curve models.",
};

interface RootLayoutProps {
  readonly children: ReactNode;
}

function readBoolean(value: string | undefined): boolean {
  return value === "true";
}

function getAuthRuntimeConfig(): AuthRuntimeConfig {
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

export default function RootLayout({ children }: RootLayoutProps): ReactElement {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <body>
        <Providers authConfig={getAuthRuntimeConfig()}>{children}</Providers>
      </body>
    </html>
  );
}
