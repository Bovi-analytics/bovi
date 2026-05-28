"use client";

import type { ReactElement } from "react";
import {
  Accordion,
  ActionIcon,
  Badge,
  Button,
  Card,
  Code,
  FileInput,
  Group,
  Select,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { Info } from "lucide-react";
import { useState } from "react";
import { downloadChallengeExport } from "@/lib/api-client";
import { MODEL_LABELS } from "@/lib/benchmark-dataset";
import { useSubmitOwnMethod } from "../hooks/use-submissions";
import type { BenchmarkModel, MilkBotRunOptions } from "@/types/api";
import { BenchmarkModelPicker } from "./benchmark-model-picker";

interface Props {
  challengeId: number;
  onSuccess: () => void;
}

const RESULTS_EXAMPLE_CSV = `TestId,LactationYield
1001,8520
1002,10450
`;

const OTHER_METHOD_VALUE = "other";
const CALCULATION_METHOD_OPTIONS = [
  { value: "Test Interval Method", label: MODEL_LABELS.tim },
  { value: "ISLC", label: MODEL_LABELS.islc },
  { value: "Best Prediction", label: MODEL_LABELS.best_predict },
  { value: "Wood", label: MODEL_LABELS.wood },
  { value: "Wilmink", label: MODEL_LABELS.wilmink },
  { value: "Ali-Schaeffer", label: MODEL_LABELS.ali_schaeffer },
  { value: "Fischer", label: MODEL_LABELS.fischer },
  { value: "MilkBot", label: MODEL_LABELS.milkbot },
  { value: "Autoencoder", label: MODEL_LABELS.autoencoder },
  { value: OTHER_METHOD_VALUE, label: "Other" },
];

function downloadText(content: string, filename: string): void {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function SubmissionFormUpload({ challengeId, onSuccess }: Props): ReactElement {
  const [file, setFile] = useState<File | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkModel>("tim");
  const [benchmarkOptions, setBenchmarkOptions] = useState<MilkBotRunOptions>({
    fitting: "frequentist",
    breed: "H",
    continent: "USA",
  });
  const [organization, setOrganization] = useState("");
  const [country, setCountry] = useState("");
  const [calcMethod, setCalcMethod] = useState<string | null>(null);
  const [otherCalcMethod, setOtherCalcMethod] = useState("");
  const { mutate, isPending, error } = useSubmitOwnMethod(challengeId);
  const calculationMethod =
    calcMethod === OTHER_METHOD_VALUE ? otherCalcMethod.trim() : (calcMethod ?? "");
  const missingOtherMethod = calcMethod === OTHER_METHOD_VALUE && !otherCalcMethod.trim();

  function handleSubmit() {
    if (!file) return;
    mutate(
      {
        file,
        meta: {
          benchmark,
          ...(benchmark === "milkbot" ? { benchmark_options: benchmarkOptions } : {}),
          organization,
          country,
          ...(calculationMethod ? { calculation_method: calculationMethod } : {}),
        },
      },
      { onSuccess }
    );
  }

  return (
    <Stack gap="md">
      <Group grow align="stretch">
        <Card withBorder padding="sm" bg="var(--mantine-color-violet-light)">
          <Stack gap={6}>
            <Group gap="xs">
              <Text size="xs" tt="uppercase" fw={700} c="violet">
                Challenger
              </Text>
              <Badge color="violet" variant="filled">
                Own method (CSV)
              </Badge>
            </Group>
            <Group gap={4} wrap="nowrap">
              <Text size="xs" c="dimmed">
                Calculate 305-day yields with your own method, then upload the results.
              </Text>
              <Tooltip label="Required columns: TestId, LactationYield. UTF-8, comma or semicolon-separated.">
                <ActionIcon size="xs" variant="subtle" color="gray">
                  <Info size={14} />
                </ActionIcon>
              </Tooltip>
            </Group>
            <Group gap="xs">
              <Button
                variant="outline"
                size="xs"
                onClick={() => void downloadChallengeExport(challengeId)}
              >
                Download test data
              </Button>
              <Button
                variant="outline"
                size="xs"
                onClick={() => downloadText(RESULTS_EXAMPLE_CSV, "results_example.csv")}
              >
                Download example results CSV
              </Button>
            </Group>
            <FileInput
              label="Upload results CSV"
              accept=".csv"
              value={file}
              onChange={setFile}
              placeholder="results.csv"
            />
            <Accordion variant="contained">
              <Accordion.Item value="format">
                <Accordion.Control>CSV format</Accordion.Control>
                <Accordion.Panel>
                  <Stack gap={6}>
                    <Text size="xs">
                      Required: <Code>TestId</Code>, <Code>LactationYield</Code>. One row per
                      lactation.
                    </Text>
                    <Code block>{RESULTS_EXAMPLE_CSV}</Code>
                  </Stack>
                </Accordion.Panel>
              </Accordion.Item>
            </Accordion>
            <Select
              label="Calculation method"
              description="Select the method used for the uploaded results."
              data={CALCULATION_METHOD_OPTIONS}
              value={calcMethod}
              onChange={setCalcMethod}
              placeholder="Select method"
              clearable
            />
            {calcMethod === OTHER_METHOD_VALUE && (
              <TextInput
                label="Other method"
                description="Please tell us which method was used."
                value={otherCalcMethod}
                onChange={(e) => setOtherCalcMethod(e.target.value)}
                required
              />
            )}
          </Stack>
        </Card>

        <Card withBorder padding="sm" bg="var(--mantine-color-blue-light)">
          <Stack gap={6}>
            <Group gap="xs">
              <Text size="xs" tt="uppercase" fw={700} c="blue">
                Benchmark
              </Text>
              <Badge color="blue" variant="filled">
                Bovi model
              </Badge>
            </Group>
            <BenchmarkModelPicker
              label="Pick a benchmark model"
              value={benchmark}
              onChange={setBenchmark}
              milkbotOptions={benchmarkOptions}
              onMilkbotOptionsChange={setBenchmarkOptions}
            />
            <Text size="xs" c="dimmed">
              Bovi runs this model server-side on the same benchmark dataset. Both your challenger
              and the benchmark are then compared against the ground-truth ALY.
            </Text>
          </Stack>
        </Card>
      </Group>

      <Group grow>
        <TextInput
          label="Organization (optional)"
          value={organization}
          onChange={(e) => setOrganization(e.target.value)}
        />
        <TextInput
          label="Country (optional)"
          value={country}
          onChange={(e) => setCountry(e.target.value)}
        />
      </Group>

      {error && (
        <Text c="red" size="xs">
          {(error as Error).message}
        </Text>
      )}

      <Button onClick={handleSubmit} loading={isPending} disabled={!file || missingOtherMethod}>
        Submit &amp; Compare
      </Button>
    </Stack>
  );
}
