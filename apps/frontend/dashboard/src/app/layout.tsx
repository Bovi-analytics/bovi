import type { ReactElement, ReactNode } from "react";
import "./globals.css";
import { Providers } from "./providers/root-provider";

export const metadata = {
  title: "Lactation Curves Dashboard",
  description: "Interactive visualization of lactation curve models.",
};

interface RootLayoutProps {
  readonly children: ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps): ReactElement {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
