"use client";

import { useState } from "react";
import type { ReactElement, ReactNode } from "react";
import { MantineProvider, createTheme } from "@mantine/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { UnitProvider } from "./unit-provider";
import { UploadedCowsProvider } from "./uploaded-cows-provider";
import { AuthProviderWrapper } from "@/lib/auth";
import type { AuthRuntimeConfig } from "@/lib/auth/config";

const theme = createTheme({
  primaryColor: "blue",
  defaultRadius: "md",
  fontFamily: "inherit",
});

interface ProvidersProps {
  readonly authConfig: AuthRuntimeConfig;
  readonly children: ReactNode;
}

export function Providers({ authConfig, children }: ProvidersProps): ReactElement {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <AuthProviderWrapper authConfig={authConfig}>
      <QueryClientProvider client={queryClient}>
        <MantineProvider theme={theme} defaultColorScheme="dark">
          <UnitProvider>
            <UploadedCowsProvider>{children}</UploadedCowsProvider>
          </UnitProvider>
        </MantineProvider>
      </QueryClientProvider>
    </AuthProviderWrapper>
  );
}
