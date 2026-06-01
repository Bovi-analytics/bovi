import type { ReactElement } from "react";
import { Loader, Stack, Text } from "@mantine/core";

export function CenteredLoader({
  label = "Loading...",
}: {
  readonly label?: string;
}): ReactElement {
  return (
    <div className="flex min-h-[50vh] items-center justify-center p-6">
      <Stack align="center" gap="sm">
        <Loader />
        <Text size="sm" c="dimmed">
          {label}
        </Text>
      </Stack>
    </div>
  );
}
