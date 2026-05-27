"use client";

import type { ReactElement } from "react";
import { Group, Stack, Text } from "@mantine/core";
import type { ChallengeDatasetSource, ChallengeDatasetStats } from "@/types/api";
import { formatDatasetSources, formatDatasetStats } from "@/lib/benchmark-dataset";

interface DatasetSummaryProps {
  readonly sources: readonly ChallengeDatasetSource[];
  readonly stats: ChallengeDatasetStats;
  readonly compact?: boolean;
}

export function DatasetSummary({
  sources,
  stats,
  compact = false,
}: DatasetSummaryProps): ReactElement {
  if (compact) {
    return (
      <Stack gap={2}>
        <Text size="xs" c="var(--benchmark-muted-text)" lineClamp={1}>
          {formatDatasetStats(stats)}
        </Text>
        <Text size="xs" c="dimmed" lineClamp={1}>
          {formatDatasetSources(sources)}
        </Text>
      </Stack>
    );
  }

  return (
    <Stack gap={6}>
      <Text size="sm" c="var(--benchmark-muted-text)">
        {formatDatasetStats(stats)}
      </Text>
      <Group gap="xs">
        {sources.map((source) => (
          <Text key={`${source.role}-${source.filename ?? source.label}`} size="xs" c="dimmed">
            {source.label}: {source.filename ?? "Unknown source"}
          </Text>
        ))}
      </Group>
    </Stack>
  );
}
