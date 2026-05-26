import type { ReactElement } from "react";
import Link from "next/link";
import { Badge, Button, Group, Stack, Text } from "@mantine/core";
import { ChevronRight, Database } from "lucide-react";
import { DataSourcePicker } from "../herd-stats/components/data-source-picker";

export default function DataUploadPage(): ReactElement {
  return (
    <div className="space-y-8 p-6">
      <Group justify="space-between" align="flex-start" gap="md">
        <div>
          <h1 className="text-2xl font-semibold">Data Upload</h1>
          <p className="mt-1 text-sm text-foreground">
            Choose the data you would like to work with. Upload your own milk recording data or
            explore the platform using one of the built-in demo herds.
          </p>
        </div>
        <Button
          component={Link}
          href="/herd-profiles"
          variant="light"
          rightSection={<ChevronRight size={14} />}
        >
          Go to Herd Profiles
        </Button>
      </Group>

      <Stack gap="md">
        <Group gap="sm" align="center">
          <Database size={18} className="text-primary" />
          <Text fw={700} size="md">
            Data source
          </Text>
          <Badge size="xs" variant="light" color="violet">
            step 1
          </Badge>
        </Group>
        <DataSourcePicker />
      </Stack>
    </div>
  );
}
