import type { ReactElement } from "react";
import Link from "next/link";
import { Badge, Button, Group, Stack, Text } from "@mantine/core";
import { ChevronRight, ClipboardList } from "lucide-react";
import { HerdProfileList } from "../herd-stats/components/herd-profile-list";

export default function HerdProfilesPage(): ReactElement {
  return (
    <div className="space-y-8 p-6">
      <Group justify="space-between" align="flex-start" gap="md">
        <div>
          <h1 className="text-2xl font-semibold">Herd Profiles</h1>
          <p className="mt-1 text-sm text-foreground">
            Save and manage the herd-level statistics used by the autoencoder. Profiles can be
            created manually or from data loaded in Data Upload.
          </p>
        </div>
        <Button
          component={Link}
          href="/curves"
          variant="light"
          rightSection={<ChevronRight size={14} />}
        >
          Continue to Curves
        </Button>
      </Group>

      <Stack gap="md">
        <Group gap="sm" align="center">
          <ClipboardList size={18} className="text-primary" />
          <Text fw={700} size="md">
            Saved profiles
          </Text>
          <Badge size="xs" variant="light" color="violet">
            step 2
          </Badge>
        </Group>
        <HerdProfileList />
      </Stack>
    </div>
  );
}
