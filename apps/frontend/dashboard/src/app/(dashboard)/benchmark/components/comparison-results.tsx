"use client";

import type { ReactElement } from "react";
import { Badge, Button, Group, SegmentedControl, Stack, Table, Text } from "@mantine/core";
import { useState } from "react";
import { downloadReportUrl } from "@/lib/api-client";
import type { SubmissionRead } from "@/types/api";

interface Props {
  submission: SubmissionRead;
}

function StatsRow({ label, stats }: { label: string; stats: Record<string, number | null> }): ReactElement {
  const fmt = (v: number | null) => (v === null ? "—" : v.toFixed(3));
  return (
    <Table.Tr>
      <Table.Td>{label}</Table.Td>
      <Table.Td>{stats.n}</Table.Td>
      <Table.Td>{fmt(stats.pearson)}</Table.Td>
      <Table.Td>{fmt(stats.rmse)} kg</Table.Td>
      <Table.Td>{fmt(stats.mae)} kg</Table.Td>
      <Table.Td>{fmt(stats.mape)}%</Table.Td>
    </Table.Tr>
  );
}

export function ComparisonResults({ submission }: Props): ReactElement {
  const [flavor, setFlavor] = useState<"icar" | "bovi" | "all">("all");
  const { stats } = submission;

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Text fw={600}>Results</Text>
        <Badge variant="light" color="green">
          {submission.submission_type === "bovi_model"
            ? `Bovi — ${submission.model_type}`
            : "Own method"}
        </Badge>
      </Group>

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
          <StatsRow label="Overall" stats={stats.overall} />
          {Object.entries(stats.by_parity).map(([p, s]) => (
            <StatsRow key={p} label={`Parity ${p}`} stats={s} />
          ))}
        </Table.Tbody>
      </Table>

      {stats.failed_count > 0 && (
        <Text size="xs" c="dimmed">{stats.failed_count} cow(s) excluded from stats.</Text>
      )}

      <Group>
        <Text size="sm" fw={500}>Report flavor:</Text>
        <SegmentedControl
          value={flavor}
          onChange={(v) => setFlavor(v as typeof flavor)}
          data={[
            { value: "icar", label: "vs ICAR" },
            { value: "bovi", label: "vs Bovi" },
            { value: "all", label: "Full" },
          ]}
          size="xs"
        />
        <Button
          size="xs"
          component="a"
          href={downloadReportUrl(submission.id, flavor)}
          download
        >
          Download PDF
        </Button>
      </Group>
    </Stack>
  );
}
