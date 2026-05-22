"use client";

import type { ReactElement } from "react";
import { Alert, Badge, Button, Group, Stack, Table, Text } from "@mantine/core";
import { downloadSubmissionReport } from "@/lib/api-client";
import type { ParityStats, SubmissionRead, VsBlock } from "@/types/api";

interface Props {
  submission: SubmissionRead;
}

function StatsRow({
  label,
  stats,
}: {
  label: string;
  stats: Record<string, number | null>;
}): ReactElement {
  const fmt = (v: number | null | undefined) =>
    v === null || v === undefined ? "-" : v.toFixed(3);
  return (
    <Table.Tr>
      <Table.Td>{label}</Table.Td>
      <Table.Td>{stats.n ?? 0}</Table.Td>
      <Table.Td>{fmt(stats.pearson)}</Table.Td>
      <Table.Td>{fmt(stats.rmse)} kg</Table.Td>
      <Table.Td>{fmt(stats.mae)} kg</Table.Td>
      <Table.Td>{fmt(stats.mape)}%</Table.Td>
    </Table.Tr>
  );
}

function StatsTable({
  title,
  caption,
  block,
}: {
  title: string;
  caption: string;
  block: VsBlock;
}): ReactElement {
  return (
    <Stack gap={4}>
      <Text fw={600} size="sm">
        {title}
      </Text>
      <Text size="xs" c="var(--benchmark-muted-text)">
        {caption}
      </Text>
      <Table withTableBorder withColumnBorders>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Group</Table.Th>
            <Table.Th>N</Table.Th>
            <Table.Th>Pearson</Table.Th>
            <Table.Th>RMSE</Table.Th>
            <Table.Th>MAE</Table.Th>
            <Table.Th>MAPE</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          <StatsRow label="Overall" stats={block.overall as ParityStats} />
          {Object.entries(block.by_parity).map(([p, s]) => (
            <StatsRow key={p} label={`Parity ${p}`} stats={s} />
          ))}
        </Table.Tbody>
      </Table>
    </Stack>
  );
}

export function ComparisonResults({ submission }: Props): ReactElement {
  const { stats } = submission;
  const challengerLabel =
    submission.submission_type === "bovi_model" ? `Bovi - ${submission.model_type}` : "Own method";
  const benchmarkLabel = submission.benchmark_model ?? "tim";

  // v2 stats: three blocks
  const isV2 =
    stats.version === 2 ||
    Boolean(stats.challenger_vs_aly || stats.benchmark_vs_aly || stats.challenger_vs_benchmark);

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Text fw={600}>Results</Text>
        <Group gap={6}>
          <Badge color="violet" variant="light">
            {challengerLabel}
          </Badge>
          <Text size="xs" c="var(--benchmark-muted-text)">
            vs
          </Text>
          <Badge color="blue" variant="light">
            {benchmarkLabel}
          </Badge>
        </Group>
      </Group>

      {isV2 ? (
        <Stack gap="md">
          {stats.challenger_vs_aly && (
            <StatsTable
              title="1. Challenger vs Actual yield (ground truth)"
              caption="The main result - how close the challenger's calculation is to ground-truth ALY."
              block={stats.challenger_vs_aly}
            />
          )}
          {stats.benchmark_vs_aly && (
            <StatsTable
              title="2. Benchmark vs Actual yield"
              caption="Baseline - how the chosen benchmark model performs against ALY on the same cohort."
              block={stats.benchmark_vs_aly}
            />
          )}
          {stats.challenger_vs_benchmark && (
            <StatsTable
              title="3. Challenger vs Benchmark"
              caption="Inter-model agreement - how the challenger and benchmark compare to each other."
              block={stats.challenger_vs_benchmark}
            />
          )}
        </Stack>
      ) : (
        <>
          <Alert color="yellow" variant="light" title="Legacy submission">
            This submission was created before the three-axis benchmarking refactor and only has the
            older vs-TIM comparison.
          </Alert>
          {stats.overall && stats.by_parity && (
            <StatsTable
              title="vs reference"
              caption="Legacy comparison."
              block={{ overall: stats.overall, by_parity: stats.by_parity }}
            />
          )}
        </>
      )}

      {stats.failed_count > 0 && (
        <Text size="xs" c="var(--benchmark-muted-text)">
          {stats.failed_count} cow(s) excluded from stats.
        </Text>
      )}

      <Group>
        <Button size="xs" onClick={() => void downloadSubmissionReport(submission.id)}>
          Download PDF
        </Button>
      </Group>
    </Stack>
  );
}
