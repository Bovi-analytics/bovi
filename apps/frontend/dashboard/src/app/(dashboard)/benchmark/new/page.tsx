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
  Table,
  Tabs,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { ArrowLeft, Info } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { getChallenge } from "@/lib/api-client";
import {
  datasetStatsFromChallengeDetail,
  formatDatasetSources,
  formatDatasetStats,
} from "@/lib/benchmark-dataset";
import type { ChallengeDatasetSource, ChallengeDatasetStats, ChallengeDetail } from "@/types/api";
import {
  useCreateChallengeFromSavedDataset,
  useCreateChallengePreset,
  useCreateChallengeUpload,
} from "../hooks/use-challenges";

interface SavedBenchmarkDataset {
  readonly id: string;
  readonly name: string;
  readonly uploadedAt: string;
  readonly rowCount: number;
  readonly cowCount: number;
  readonly actualYieldCount: number;
  readonly datasetSources: ChallengeDatasetSource[];
  readonly datasetStats: ChallengeDatasetStats;
  readonly cowMetadata: ChallengeDetail["cow_metadata"];
  readonly actualYields: NonNullable<ChallengeDetail["actual_yields"]>;
}

const SAVED_BENCHMARK_DATASETS_KEY = "bovi-saved-benchmark-datasets-v1";

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

function loadSavedBenchmarkDatasets(): SavedBenchmarkDataset[] {
  try {
    const stored = localStorage.getItem(SAVED_BENCHMARK_DATASETS_KEY);
    if (!stored) return [];
    const parsed = JSON.parse(stored) as SavedBenchmarkDataset[];
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item) => item?.cowMetadata && item?.actualYields)
      .map((item) => ({
        ...item,
        datasetSources: item.datasetSources ?? [],
        datasetStats:
          item.datasetStats ?? datasetStatsFromChallengeDetail(item.cowMetadata, item.actualYields),
      }));
  } catch {
    localStorage.removeItem(SAVED_BENCHMARK_DATASETS_KEY);
    return [];
  }
}

function saveBenchmarkDataset(item: SavedBenchmarkDataset): SavedBenchmarkDataset[] {
  const next = [
    item,
    ...loadSavedBenchmarkDatasets().filter((saved) => saved.id !== item.id),
  ].slice(0, 10);
  localStorage.setItem(SAVED_BENCHMARK_DATASETS_KEY, JSON.stringify(next));
  return next;
}

function PresetTab({ onCreated }: { onCreated: (id: number) => void }): ReactElement {
  const [name, setName] = useState("");
  const { mutate, isPending, error } = useCreateChallengePreset();

  return (
    <Stack gap="md">
      <Card withBorder padding="md">
        <Stack gap="sm">
          <Text fw={600}>Demo dataset</Text>
          <Text size="sm" c="var(--benchmark-muted-text)">
            407 lactations with sparse test-day records and ground-truth Actual Lactation Yield
            (ALY) from daily milk meter recordings. This benchmark demo dataset is separate from
            Demo herd A and Demo herd B in Data Upload.
          </Text>
          <Text size="xs" c="dimmed">
            Test-day records: TestDataSet.csv · Ground-truth ALY: ActualMilkYields.csv
          </Text>
          <TextInput
            label="Challenge name (optional)"
            placeholder="Demo dataset"
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
            Create demo dataset challenge
          </Button>
        </Stack>
      </Card>
    </Stack>
  );
}

function UploadTab({
  onCreated,
  onSaved,
}: {
  onCreated: (id: number) => void;
  onSaved: (items: SavedBenchmarkDataset[]) => void;
}): ReactElement {
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
        label="Challenge name"
        placeholder="My farm lactations 2025"
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
            <Tooltip label="Sparse test-day milk recordings, multiple rows per lactation.">
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
                    Optional: <Code>parity</Code>. One row per test-day; multiple rows per
                    lactation. Comma or semicolon-separated, UTF-8.
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
            {
              onSuccess: async (c) => {
                const detail = await getChallenge(c.id);
                if (detail.actual_yields) {
                  onSaved(
                    saveBenchmarkDataset({
                      id: `${Date.now()}-${name.trim()}`,
                      name: name.trim(),
                      uploadedAt: new Date().toISOString(),
                      rowCount: Object.values(detail.cow_metadata).reduce(
                        (total, cow) => total + cow.dim.length,
                        0
                      ),
                      cowCount: Object.keys(detail.cow_metadata).length,
                      actualYieldCount: Object.keys(detail.actual_yields).length,
                      datasetSources: detail.dataset_sources,
                      datasetStats: detail.dataset_stats,
                      cowMetadata: detail.cow_metadata,
                      actualYields: detail.actual_yields,
                    })
                  );
                }
                onCreated(c.id);
              },
            }
          );
        }}
      >
        Create challenge
      </Button>
    </Stack>
  );
}

function SavedDatasetTab({
  datasets,
  onCreated,
}: {
  datasets: SavedBenchmarkDataset[];
  onCreated: (id: number) => void;
}): ReactElement {
  const [selectedId, setSelectedId] = useState<string | null>(datasets[0]?.id ?? null);
  const [name, setName] = useState("");
  const { mutate, isPending, error } = useCreateChallengeFromSavedDataset();
  const selected = datasets.find((item) => item.id === selectedId) ?? null;

  useEffect(() => {
    if (!selectedId && datasets[0]) setSelectedId(datasets[0].id);
  }, [datasets, selectedId]);

  if (datasets.length === 0) {
    return (
      <Alert color="gray" variant="light">
        Upload an own benchmark dataset once to make it available here later.
      </Alert>
    );
  }

  return (
    <Stack gap="md">
      <TextInput
        label="Challenge name"
        placeholder={selected ? selected.name : "Saved benchmark dataset"}
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <Table striped highlightOnHover withColumnBorders fz="sm">
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Status</Table.Th>
            <Table.Th>Dataset</Table.Th>
            <Table.Th>Uploaded</Table.Th>
            <Table.Th>Rows</Table.Th>
            <Table.Th>Lactations</Table.Th>
            <Table.Th>ALY rows</Table.Th>
            <Table.Th>Sources</Table.Th>
            <Table.Th />
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {datasets.map((dataset) => {
            const active = dataset.id === selectedId;
            return (
              <Table.Tr key={dataset.id}>
                <Table.Td>{active ? "Selected" : "Saved"}</Table.Td>
                <Table.Td>{dataset.name}</Table.Td>
                <Table.Td>{new Date(dataset.uploadedAt).toLocaleString()}</Table.Td>
                <Table.Td>{dataset.rowCount.toLocaleString()}</Table.Td>
                <Table.Td>{dataset.cowCount.toLocaleString()}</Table.Td>
                <Table.Td>{dataset.actualYieldCount.toLocaleString()}</Table.Td>
                <Table.Td>
                  <Text size="xs" c="var(--benchmark-muted-text)">
                    {formatDatasetStats(dataset.datasetStats)}
                  </Text>
                  <Text size="xs" c="dimmed" lineClamp={1}>
                    {formatDatasetSources(dataset.datasetSources)}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Button
                    size="xs"
                    variant={active ? "light" : "filled"}
                    color="violet"
                    disabled={active}
                    onClick={() => setSelectedId(dataset.id)}
                  >
                    {active ? "Selected" : "Select"}
                  </Button>
                </Table.Td>
              </Table.Tr>
            );
          })}
        </Table.Tbody>
      </Table>

      {error && (
        <Text c="red" size="sm">
          {(error as Error).message}
        </Text>
      )}

      <Button
        disabled={!selected}
        loading={isPending}
        w="fit-content"
        onClick={() => {
          if (!selected) return;
          mutate(
            {
              name: name.trim() || selected.name,
              cowMetadata: selected.cowMetadata,
              actualYields: selected.actualYields,
              datasetSources: selected.datasetSources,
            },
            { onSuccess: (c) => onCreated(c.id) }
          );
        }}
      >
        Create challenge from saved dataset
      </Button>
    </Stack>
  );
}

export default function NewChallengePage(): ReactElement {
  const router = useRouter();
  const [savedDatasets, setSavedDatasets] = useState<SavedBenchmarkDataset[]>([]);

  useEffect(() => {
    setSavedDatasets(loadSavedBenchmarkDatasets());
  }, []);

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
          Pick a benchmark dataset with ground-truth Actual Lactation Yield (ALY). Use the built-in
          demo dataset, upload your own test-day records with daily milk meter ground truth, or
          select a saved own dataset.
        </Text>
      </Stack>

      <Tabs defaultValue="preset">
        <Tabs.List>
          <Tabs.Tab value="preset">Demo dataset</Tabs.Tab>
          <Tabs.Tab value="upload">Upload own dataset</Tabs.Tab>
          {savedDatasets.length > 0 && <Tabs.Tab value="saved">Select dataset</Tabs.Tab>}
        </Tabs.List>
        <Tabs.Panel value="preset" pt="md">
          <PresetTab onCreated={onCreated} />
        </Tabs.Panel>
        <Tabs.Panel value="upload" pt="md">
          <UploadTab onCreated={onCreated} onSaved={setSavedDatasets} />
        </Tabs.Panel>
        {savedDatasets.length > 0 && (
          <Tabs.Panel value="saved" pt="md">
            <SavedDatasetTab datasets={savedDatasets} onCreated={onCreated} />
          </Tabs.Panel>
        )}
      </Tabs>

      <Group gap="xs">
        <Loader size="xs" style={{ display: "none" }} />
        <Text size="xs" c="var(--benchmark-muted-text)">
          Creating an upload challenge takes a few seconds. Once created, run the challenger and
          benchmark on the selected dataset to see results.
        </Text>
      </Group>
    </div>
  );
}
