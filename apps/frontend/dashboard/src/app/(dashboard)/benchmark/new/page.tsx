"use client";

import type { ReactElement } from "react";
import {
  Accordion,
  ActionIcon,
  Alert,
  Anchor,
  Button,
  Card,
  Code,
  FileInput,
  Group,
  Loader,
  Stack,
  Tabs,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { ArrowLeft, Info } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useCreateChallengePreset, useCreateChallengeUpload } from "../hooks/use-challenges";

const TEST_DAY_EXAMPLE_CSV = `TestId,parity,dim,milk_kg
1001,1,15,28.5
1001,1,45,32.1
1001,1,90,30.4
1001,1,180,24.8
1001,1,270,18.2
1002,3,12,40.0
1002,3,60,38.5
1002,3,150,30.0
`;

const ACTUAL_YIELDS_EXAMPLE_CSV = `TestId,LactationYield
1001,8520
1002,10450
`;

function downloadText(content: string, filename: string): void {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function PresetTab({ onCreated }: { onCreated: (id: number) => void }): ReactElement {
  const [name, setName] = useState("");
  const { mutate, isPending, error } = useCreateChallengePreset();

  return (
    <Stack gap="md">
      <Card withBorder padding="md">
        <Stack gap="sm">
          <Text fw={600}>Reference dataset</Text>
          <Text size="sm" c="var(--benchmark-muted-text)">
            407 cows with sparse test-day records and ground-truth Actual Lactation Yield (ALY) from
            daily milk meter recordings. This is the built-in dataset that can be used for
            ground-truth benchmarking.
          </Text>
          <TextInput
            label="Cohort name (optional)"
            placeholder="Reference dataset"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          {error && (
            <Text c="red" size="sm">
              {(error as Error).message}
            </Text>
          )}
          <Button
            onClick={() =>
              mutate(
                { source: "preset", preset: "icar", name: name || undefined },
                { onSuccess: (c) => onCreated(c.id) }
              )
            }
            loading={isPending}
            w="fit-content"
          >
            Create reference dataset challenge
          </Button>
        </Stack>
      </Card>
    </Stack>
  );
}

function UploadTab({ onCreated }: { onCreated: (id: number) => void }): ReactElement {
  const [name, setName] = useState("");
  const [testDayCsv, setTestDayCsv] = useState<File | null>(null);
  const [actualYieldsCsv, setActualYieldsCsv] = useState<File | null>(null);
  const { mutate, isPending, error } = useCreateChallengeUpload();

  const canSubmit = name.trim() && testDayCsv && actualYieldsCsv;

  return (
    <Stack gap="md">
      <Alert color="blue" variant="light" title="Why both files?">
        Without ground-truth daily milk meter yields we can&apos;t validate any calculation. Both
        the test-day records <em>and</em> the actual-yields CSV are required.
      </Alert>

      <TextInput
        label="Cohort name"
        placeholder="My farm cohort 2025"
        required
        value={name}
        onChange={(e) => setName(e.target.value)}
      />

      <Card withBorder padding="sm">
        <Stack gap={6}>
          <Group gap={6}>
            <Text fw={600} size="sm">
              Test-day records CSV
            </Text>
            <Tooltip label="Sparse test-day milk recordings, multiple rows per cow.">
              <ActionIcon size="xs" variant="subtle" color="gray">
                <Info size={14} />
              </ActionIcon>
            </Tooltip>
          </Group>
          <FileInput
            accept=".csv"
            value={testDayCsv}
            onChange={setTestDayCsv}
            placeholder="test_day.csv"
          />
          <Accordion variant="contained">
            <Accordion.Item value="format">
              <Accordion.Control>CSV format</Accordion.Control>
              <Accordion.Panel>
                <Stack gap={6}>
                  <Text size="xs">
                    Required columns: <Code>TestId</Code>, <Code>dim</Code>, <Code>milk_kg</Code>.
                    Optional: <Code>parity</Code>. One row per test-day; multiple rows per cow.
                    Comma or semicolon-separated, UTF-8.
                  </Text>
                  <Code block>{TEST_DAY_EXAMPLE_CSV}</Code>
                  <Button
                    variant="outline"
                    size="xs"
                    w="fit-content"
                    onClick={() => downloadText(TEST_DAY_EXAMPLE_CSV, "test_day_example.csv")}
                  >
                    Download example CSV
                  </Button>
                </Stack>
              </Accordion.Panel>
            </Accordion.Item>
          </Accordion>
        </Stack>
      </Card>

      <Card withBorder padding="sm">
        <Stack gap={6}>
          <Group gap={6}>
            <Text fw={600} size="sm">
              Actual yields CSV (ground truth)
            </Text>
            <Tooltip label="Daily milk meter cumulative 305-day yield per lactation.">
              <ActionIcon size="xs" variant="subtle" color="gray">
                <Info size={14} />
              </ActionIcon>
            </Tooltip>
          </Group>
          <FileInput
            accept=".csv"
            value={actualYieldsCsv}
            onChange={setActualYieldsCsv}
            placeholder="actual_yields.csv"
          />
          <Accordion variant="contained">
            <Accordion.Item value="format">
              <Accordion.Control>CSV format</Accordion.Control>
              <Accordion.Panel>
                <Stack gap={6}>
                  <Text size="xs">
                    Required columns: <Code>TestId</Code>, <Code>LactationYield</Code>. One row per
                    lactation.
                  </Text>
                  <Code block>{ACTUAL_YIELDS_EXAMPLE_CSV}</Code>
                  <Button
                    variant="outline"
                    size="xs"
                    w="fit-content"
                    onClick={() =>
                      downloadText(ACTUAL_YIELDS_EXAMPLE_CSV, "actual_yields_example.csv")
                    }
                  >
                    Download example CSV
                  </Button>
                </Stack>
              </Accordion.Panel>
            </Accordion.Item>
          </Accordion>
        </Stack>
      </Card>

      {error && (
        <Text c="red" size="sm">
          {(error as Error).message}
        </Text>
      )}

      <Button
        disabled={!canSubmit}
        loading={isPending}
        w="fit-content"
        onClick={() => {
          if (!testDayCsv || !actualYieldsCsv) return;
          mutate(
            { name: name.trim(), testDayCsv, actualYieldsCsv },
            { onSuccess: (c) => onCreated(c.id) }
          );
        }}
      >
        Create challenge
      </Button>
    </Stack>
  );
}

export default function NewChallengePage(): ReactElement {
  const router = useRouter();
  const onCreated = (id: number) => router.push(`/benchmark/${id}`);

  return (
    <div className="benchmark-page max-w-2xl space-y-6 p-6">
      <Anchor component={Link} href="/benchmark" size="sm">
        <Group gap={4} wrap="nowrap">
          <ArrowLeft size={14} />
          <span>Back to all challenges</span>
        </Group>
      </Anchor>
      <Stack gap={4}>
        <h1 className="text-2xl font-semibold">New Challenge</h1>
        <Text size="sm" c="var(--benchmark-muted-text)">
          Pick a cohort with ground-truth Actual Lactation Yield (ALY). Use the built-in reference
          dataset, or upload your own test-day records together with daily milk meter ground truth.
        </Text>
      </Stack>

      <Tabs defaultValue="preset">
        <Tabs.List>
          <Tabs.Tab value="preset">Reference dataset</Tabs.Tab>
          <Tabs.Tab value="upload">Upload own dataset</Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="preset" pt="md">
          <PresetTab onCreated={onCreated} />
        </Tabs.Panel>
        <Tabs.Panel value="upload" pt="md">
          <UploadTab onCreated={onCreated} />
        </Tabs.Panel>
      </Tabs>

      <Group gap="xs">
        <Loader size="xs" style={{ display: "none" }} />
        <Text size="xs" c="var(--benchmark-muted-text)">
          Creating an upload challenge takes a few seconds. Once created, run the challenger and
          benchmark on the cohort to see results.
        </Text>
      </Group>
    </div>
  );
}
