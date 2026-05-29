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
  const clientId = process.env["AZURE_AD_CLIENT_ID"] ?? "";
  const configuredScope = process.env["AZURE_AD_API_SCOPE"];
  return {
    clientId,
    apiScope: configuredScope ?? (clientId ? `api://${clientId}/access_as_user` : ""),
    authDisabled: readBoolean(process.env["AUTH_DISABLED"]),
    redirectUri: process.env["AUTH_REDIRECT_URI"] ?? "/auth/login",
    postLogoutRedirectUri: process.env["AUTH_POST_LOGOUT_REDIRECT_URI"] ?? "/",
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
