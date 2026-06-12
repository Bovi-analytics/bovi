"use client";

import { useState } from "react";
import type { ReactElement, ReactNode } from "react";
import { MantineProvider, createTheme, localStorageColorSchemeManager } from "@mantine/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { UnitProvider } from "./unit-provider";
import { UploadedCowsProvider } from "./uploaded-cows-provider";
import { AuthProviderWrapper } from "@/lib/auth";
import type { AuthRuntimeConfig } from "@/lib/auth/config";
import { DEFAULT_THEME, THEME_STORAGE_KEY } from "@/lib/theme";

const theme = createTheme({
  primaryColor: "blue",
  defaultRadius: "md",
  fontFamily: "inherit",
});

const colorSchemeManager = localStorageColorSchemeManager({
  key: THEME_STORAGE_KEY,
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
        <MantineProvider
          theme={theme}
          colorSchemeManager={colorSchemeManager}
          defaultColorScheme={DEFAULT_THEME}
        >
          <UnitProvider>
            <UploadedCowsProvider>{children}</UploadedCowsProvider>
          </UnitProvider>
        </MantineProvider>
      </QueryClientProvider>
    </AuthProviderWrapper>
  );
}
