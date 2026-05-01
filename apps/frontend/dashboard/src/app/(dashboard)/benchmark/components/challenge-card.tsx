"use client";

import type { ReactElement } from "react";
import { Badge, Button, Card, Group, Stack, Text } from "@mantine/core";
import { useRouter } from "next/navigation";
import type { ChallengeRead } from "@/types/api";

interface Props {
  challenge: ChallengeRead;
}

export function ChallengeCard({ challenge }: Props): ReactElement {
  const router = useRouter();
  return (
    <Card shadow="sm" padding="md" radius="md" withBorder>
      <Stack gap="xs">
        <Group justify="space-between">
          <Text fw={600} size="sm">Challenge #{challenge.id}</Text>
          <Badge size="xs" variant="light">{challenge.dataset}</Badge>
        </Group>
        <Text size="xs" c="dimmed">
          {challenge.size} · {challenge.period}
          {challenge.created_at
            ? ` · ${new Date(challenge.created_at).toLocaleDateString()}`
            : ""}
        </Text>
        <Button
          size="xs"
          variant="light"
          onClick={() => router.push(`/benchmark/${challenge.id}`)}
        >
          View &amp; Submit
        </Button>
      </Stack>
    </Card>
  );
}
