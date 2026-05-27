"use client";

import { useState } from "react";
import type { ReactElement, ReactNode } from "react";
import { MantineProvider, createTheme } from "@mantine/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { UnitProvider } from "./unit-provider";
import { UploadedCowsProvider } from "./uploaded-cows-provider";

const theme = createTheme({
  primaryColor: "blue",
  defaultRadius: "md",
  fontFamily: "inherit",
});

interface ProvidersProps {
  readonly children: ReactNode;
}

export function Providers({ children }: ProvidersProps): ReactElement {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      <MantineProvider theme={theme} defaultColorScheme="dark">
        <UnitProvider>
          <UploadedCowsProvider>{children}</UploadedCowsProvider>
        </UnitProvider>
      </MantineProvider>
    </QueryClientProvider>
  );
}
