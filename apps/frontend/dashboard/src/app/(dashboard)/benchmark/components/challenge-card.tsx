"use client";

import type { ReactElement } from "react";
import { Badge, Button, Card, Group, Stack, Text } from "@mantine/core";
import { useRouter } from "next/navigation";
import type { ChallengeRead } from "@/types/api";

interface Props {
  challenge: ChallengeRead;
}

const DATASET_LABEL: Record<string, string> = {
  icar: "Reference cohort",
  upload: "Custom upload",
};

export function ChallengeCard({ challenge }: Props): ReactElement {
  const router = useRouter();
  const title = challenge.name ?? `Challenge #${challenge.id}`;
  const datasetLabel = DATASET_LABEL[challenge.dataset] ?? challenge.dataset;
  return (
    <Card shadow="sm" padding="md" radius="md" withBorder>
      <Stack gap="xs">
        <Group justify="space-between">
          <Text fw={600} size="sm">
            {title}
          </Text>
          <Badge size="xs" variant="light">
            {datasetLabel}
          </Badge>
        </Group>
        <Text size="xs" c="var(--benchmark-muted-text)">
          #{challenge.id}
          {challenge.created_at ? ` · ${new Date(challenge.created_at).toLocaleDateString()}` : ""}
        </Text>
        <Button size="xs" variant="light" onClick={() => router.push(`/benchmark/${challenge.id}`)}>
          View &amp; Submit
        </Button>
      </Stack>
    </Card>
  );
}
