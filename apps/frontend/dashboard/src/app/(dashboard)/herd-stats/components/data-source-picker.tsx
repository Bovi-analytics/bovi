"use client";

import { useEffect, useRef, useState } from "react";
import type { ReactElement } from "react";
import {
  Accordion,
  Alert,
  Badge,
  Button,
  Code,
  Group,
  Loader,
  Modal,
  Paper,
  SegmentedControl,
  Stack,
  Table,
  Text,
  UnstyledButton,
} from "@mantine/core";
import { AlertCircle, CheckCircle2, ChevronRight, Download } from "lucide-react";
import Link from "next/link";
import { HERD_STATS_METADATA } from "@/data/herd-stats-metadata";
import { statsToHerdProfileFields } from "@/lib/herd-profile-utils";
import { useUploadedCows } from "@/app/providers/uploaded-cows-provider";
import { usePresetDataset } from "@/app/(dashboard)/curves/hooks/use-preset-dataset";
import type {
  HerdProfileUploadResponse,
  PresetDatasetKey,
  PresetPeriodKey,
  PresetSizeKey,
} from "@/types/api";
import { HerdProfileForm } from "./herd-profile-form";
import { useHerdProfileUpload } from "../hooks/use-herd-profile-upload";
import { useCreateHerdProfile } from "../hooks/use-herd-profiles";

/* ------------------------------------------------------------------ */
/*  Types & constants                                                  */
/* ------------------------------------------------------------------ */

type SourceKey = PresetDatasetKey | "upload";
type FormatKey = "aggregated" | "icar_test_day" | "dairycom_test_day";

interface SourceOption {
  value: SourceKey;
  label: string;
  description: string;
}

const SOURCE_OPTIONS: SourceOption[] = [
  {
    value: "aurora",
    label: "Aurora Ridge",
    description: "5,102 cows · US dairy · 2023–2025",
  },
  {
    value: "sunnyside",
    label: "Sunnyside",
    description: "1,000+ cows · US dairy · 2000–2026",
  },
  {
    value: "upload",
    label: "Upload a file",
    description: "Bring your own CSV from farm software",
  },
];

const SIZE_OPTIONS = [
  { value: "small", label: "Small (~200)" },
  { value: "medium", label: "Medium (~1k)" },
  { value: "large", label: "Large (all)" },
];

const PERIOD_OPTIONS = [
  { value: "recent", label: "Recent" },
  { value: "old", label: "Old" },
  { value: "mixed", label: "Mixed" },
];

interface FormatMeta {
  label: string;
  blurb: string;
  columns: Array<{ name: string; description: string; required: boolean }>;
  template: () => string;
  templateName: string;
}

const ICAR_TEMPLATE =
  "TestId,TestDate,EventType,CalvingDate,BirthDate,Parity,DaysInMilk,DailyMilkingYield\n" +
  "1483,6/18/2019,MilkRecording,6/3/2019,2/9/2009,7,15,49.1\n" +
  "1483,7/16/2019,MilkRecording,6/3/2019,2/9/2009,7,43,53.4\n" +
  "1483,8/13/2019,MilkRecording,6/3/2019,2/9/2009,7,71,52.1\n" +
  "1528,6/18/2019,MilkRecording,6/5/2019,1/3/2011,5,13,45.8\n";

const DAIRYCOM_TEMPLATE =
  '"ID";"TestDate";"DIM";"MILK";"PCTF";"PCTP";"FCM";"305ME";"RELV";"SCC";"LS";"PEN";\n' +
  "     407 ;09/27/24; 181 ; 97  ;  3,1 ;  3,0 ; 91 ;29920 ;  97 ;   22 ;0,8 ;  6 ;\n" +
  "     512 ;09/27/24;  42 ;110  ;  3,2 ;  3,0 ;107 ;28500 ;  95 ;   18 ;0,6 ;  6 ;\n";

function buildAggregatedTemplate(): string {
  const headers = HERD_STATS_METADATA.map((m) => m.name).join(",");
  const exampleRow = HERD_STATS_METADATA.map((m) => {
    const mid = (m.rawMin + m.rawMax) / 2;
    return Math.round(mid * 100) / 100;
  }).join(",");
  return `${headers}\n${exampleRow}\n`;
}

const FORMATS: Record<FormatKey, FormatMeta> = {
  aggregated: {
    label: "Herd summary",
    blurb:
      "A single row with the 10 statistics already averaged across your herd. Use this when your farm software exports a per-herd aggregate rather than individual cow records.",
    columns: HERD_STATS_METADATA.map((m) => ({
      name: m.name,
      description: `${m.description} (${m.unit || "0–1 score"}, typical ${m.rawMin}–${m.rawMax})`,
      required: false,
    })),
    template: buildAggregatedTemplate,
    templateName: "herd_stats_template_aggregated.csv",
  },
  icar_test_day: {
    label: "Milk recordings",
    blurb:
      "One row per cow per recording date — the raw export you get from milk recording software. We calculate herd averages from these records automatically. Also extracts individual cow data for use on the Curves tab.",
    columns: [
      { name: "TestId", description: "Unique cow identifier", required: true },
      { name: "DaysInMilk", description: "Days since calving", required: true },
      { name: "DailyMilkingYield", description: "Daily milk yield in kg", required: true },
      { name: "Parity", description: "Lactation number", required: false },
      { name: "EventType", description: "Only MilkRecording rows are used", required: false },
    ],
    template: () => ICAR_TEMPLATE,
    templateName: "herd_stats_template_icar.csv",
  },
  dairycom_test_day: {
    label: "DairyCom export",
    blurb:
      "Semicolon-separated export from DairyCom (Cornell-style software). Milk values in lbs are converted to kg automatically. European decimal notation (3,1) is handled.",
    columns: [
      { name: "ID", description: "Unique cow identifier", required: true },
      { name: "DIM", description: "Days in milk for this test day", required: true },
      { name: "MILK", description: "Daily milk yield in lbs (auto-converted to kg)", required: true },
      { name: "305ME", description: "305-day mature equivalent in lbs", required: false },
    ],
    template: () => DAIRYCOM_TEMPLATE,
    templateName: "herd_stats_template_dairycom.csv",
  },
};

function downloadText(content: string, filename: string): void {
  const blob = new Blob([content], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/* ------------------------------------------------------------------ */
/*  Preset panel                                                       */
/* ------------------------------------------------------------------ */

function PresetPanel({ dataset }: { dataset: PresetDatasetKey }): ReactElement {
  const { activePreset, setActivePreset } = useUploadedCows();

  // Local state so controls work before the dataset is activated
  const [selectedSize, setSelectedSize] = useState<PresetSizeKey>(
    activePreset?.dataset === dataset ? activePreset.size : "small"
  );
  const [selectedPeriod, setSelectedPeriod] = useState<PresetPeriodKey>(
    activePreset?.dataset === dataset ? activePreset.period : "mixed"
  );

  // Always fetch based on local selection → user gets a live preview before clicking "Use"
  const { data: presetData, isLoading, isError } = usePresetDataset(
    dataset,
    selectedSize,
    selectedPeriod
  );

  const isActive =
    activePreset?.dataset === dataset &&
    activePreset.size === selectedSize &&
    activePreset.period === selectedPeriod;

  function activate() {
    setActivePreset({ dataset, size: selectedSize, period: selectedPeriod });
  }

  return (
    <Stack gap="md">
      <Stack gap="xs">
        <Text size="xs" c="dimmed">Sample size</Text>
        <SegmentedControl
          size="xs"
          value={selectedSize}
          onChange={(v) => setSelectedSize(v as PresetSizeKey)}
          data={SIZE_OPTIONS}
        />
      </Stack>
      <Stack gap="xs">
        <Text size="xs" c="dimmed">Time period</Text>
        <SegmentedControl
          size="xs"
          value={selectedPeriod}
          onChange={(v) => setSelectedPeriod(v as PresetPeriodKey)}
          data={PERIOD_OPTIONS}
        />
      </Stack>

      <Group gap="sm" align="center">
        {isLoading && <Loader size="xs" />}
        {presetData && !isLoading && (
          <Badge color="violet" variant="light">
            {presetData.cow_count.toLocaleString()} cows
          </Badge>
        )}
        {!isActive && (
          <Button
            size="sm"
            color="violet"
            variant="outline"
            onClick={activate}
            disabled={isLoading || !presetData}
          >
            Use this dataset
          </Button>
        )}
      </Group>

      {isActive && (
        <Alert color="violet" variant="light" p="sm">
          <Group justify="space-between" align="center">
            <Text size="sm">
              Dataset active — {presetData?.cow_count.toLocaleString()} cows ready.
            </Text>
            <Button
              component={Link}
              href="/curves"
              size="xs"
              color="violet"
              rightSection={<ChevronRight size={14} />}
            >
              Go to Curves
            </Button>
          </Group>
        </Alert>
      )}

      {isError && (
        <Alert icon={<AlertCircle size={14} />} color="red" p="xs">
          Dataset unavailable — make sure CONNECTION_STRING is configured and the preprocessing
          script has been run.
        </Alert>
      )}
    </Stack>
  );
}

/* ------------------------------------------------------------------ */
/*  Upload panel                                                       */
/* ------------------------------------------------------------------ */

function UploadPanel(): ReactElement {
  const [selectedFormat, setSelectedFormat] = useState<FormatKey>("icar_test_day");
  const [preview, setPreview] = useState<HerdProfileUploadResponse | null>(null);
  const [saveOpen, setSaveOpen] = useState(false);
  const [uploadedFilename, setUploadedFilename] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const uploadMutation = useHerdProfileUpload();
  const createMutation = useCreateHerdProfile();
  const { setDataset } = useUploadedCows();

  const activeFormat = FORMATS[selectedFormat];
  const detectedMismatch = preview !== null && preview.format_detected !== selectedFormat;

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const filename = file.name;
    uploadMutation.mutate(file, {
      onSuccess: (response) => {
        setPreview(response);
        setUploadedFilename(filename);
        if (
          response.cows.length > 0 &&
          (response.format_detected === "icar_test_day" ||
            response.format_detected === "dairycom_test_day")
        ) {
          setDataset({
            name: filename,
            format: response.format_detected,
            uploadedAt: new Date().toISOString(),
            cows: response.cows.map((c) => ({
              cowId: c.cow_id,
              parity: c.parity,
              dim: c.dim,
              milkrecordings: c.milk_kg,
            })),
          });
        }
      },
    });
    e.target.value = "";
  }

  function getPreviewStatsArray(): number[] {
    if (!preview) return [];
    return HERD_STATS_METADATA.map((meta) => preview.stats[meta.name] ?? meta.default);
  }

  const FORMAT_LABELS: Record<FormatKey, string> = {
    aggregated: FORMATS.aggregated.label,
    icar_test_day: FORMATS.icar_test_day.label,
    dairycom_test_day: FORMATS.dairycom_test_day.label,
  };

  return (
    <Stack gap="md">
      <Stack gap="xs">
        <Text size="xs" c="dimmed">File format</Text>
        <SegmentedControl
          size="xs"
          value={selectedFormat}
          onChange={(v) => setSelectedFormat(v as FormatKey)}
          data={(Object.keys(FORMATS) as FormatKey[]).map((k) => ({
            value: k,
            label: FORMATS[k].label,
          }))}
        />
        <Text size="xs" c="dimmed">{activeFormat.blurb}</Text>
      </Stack>

      <Accordion variant="contained">
        <Accordion.Item value="columns">
          <Accordion.Control>
            <Text size="xs" fw={500}>Expected columns</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Stack gap="xs">
              <Table striped withColumnBorders fz="xs">
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Column</Table.Th>
                    <Table.Th>Description</Table.Th>
                    <Table.Th>Required</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {activeFormat.columns.map((c) => (
                    <Table.Tr key={c.name}>
                      <Table.Td><Code>{c.name}</Code></Table.Td>
                      <Table.Td>{c.description}</Table.Td>
                      <Table.Td>
                        {c.required ? (
                          <Badge size="xs" color="red" variant="light">required</Badge>
                        ) : (
                          <Text size="xs" c="dimmed">optional</Text>
                        )}
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
              <Button
                size="xs"
                variant="subtle"
                leftSection={<Download size={14} />}
                style={{ width: "fit-content" }}
                onClick={() => downloadText(activeFormat.template(), activeFormat.templateName)}
              >
                Download example CSV
              </Button>
            </Stack>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <input
        ref={inputRef}
        type="file"
        accept=".csv,.CSV"
        style={{ display: "none" }}
        onChange={handleFileChange}
      />
      <Button
        variant="outline"
        size="sm"
        onClick={() => inputRef.current?.click()}
        loading={uploadMutation.isPending}
        style={{ width: "fit-content" }}
      >
        Choose CSV file…
      </Button>

      {uploadMutation.isError && (
        <Alert icon={<AlertCircle size={16} />} color="red" title="Upload failed">
          {(uploadMutation.error as Error).message}
        </Alert>
      )}

      {preview && (
        <Stack gap="sm">
          {detectedMismatch && (
            <Alert icon={<AlertCircle size={14} />} color="blue">
              Detected format: <Code>{FORMAT_LABELS[preview.format_detected]}</Code>. Switched
              from <Code>{FORMAT_LABELS[selectedFormat]}</Code>.
            </Alert>
          )}
          <Group gap="xs">
            <Badge variant="light">{FORMAT_LABELS[preview.format_detected]}</Badge>
            <Text size="xs" c="dimmed">
              {preview.row_count.toLocaleString()} row(s)
              {preview.cow_count != null && ` · ${preview.cow_count} cows`}
              {preview.detected_parity != null && ` · dominant parity ${preview.detected_parity}`}
            </Text>
          </Group>
          {preview.warnings.map((w, i) => (
            <Alert key={i} icon={<AlertCircle size={14} />} color="yellow">{w}</Alert>
          ))}
          {preview.cows.length > 0 && uploadedFilename && (
            <Alert icon={<CheckCircle2 size={14} />} color="green">
              {preview.cows.length} cow records from <Code>{uploadedFilename}</Code> ready for the{" "}
              <Link href="/curves" style={{ textDecoration: "underline" }}>Curves tab</Link>.
            </Alert>
          )}
          <Table striped withColumnBorders fz="xs">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Stat</Table.Th>
                <Table.Th>Raw value</Table.Th>
                <Table.Th>Normalised (0–1)</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {HERD_STATS_METADATA.map((meta) => {
                const filled = preview.stats[meta.name] !== undefined;
                const raw = preview.raw_stats[meta.name];
                const unit = meta.unit || "";
                const rawDigits = raw !== undefined && Math.abs(raw) >= 100 ? 0 : 2;
                return (
                  <Table.Tr key={meta.name}>
                    <Table.Td>{meta.label}</Table.Td>
                    <Table.Td>
                      {raw !== undefined ? (
                        `${raw.toFixed(rawDigits)}${unit ? ` ${unit}` : ""}`
                      ) : (
                        <Text size="xs" c="dimmed">—</Text>
                      )}
                    </Table.Td>
                    <Table.Td>
                      {filled ? (
                        preview.stats[meta.name]?.toFixed(3)
                      ) : (
                        <Text size="xs" c="dimmed">slider default</Text>
                      )}
                    </Table.Td>
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
          <Group>
            <Button size="sm" color="violet" onClick={() => setSaveOpen(true)}>
              Save as profile…
            </Button>
            <Button size="sm" variant="subtle" onClick={() => setPreview(null)}>
              Discard
            </Button>
          </Group>
        </Stack>
      )}

      <Modal
        opened={saveOpen}
        onClose={() => setSaveOpen(false)}
        title="Save herd profile"
        size="xl"
      >
        <HerdProfileForm
          initial={
            preview
              ? {
                  id: -1,
                  name: "",
                  description: "",
                  created_at: null,
                  updated_at: null,
                  ...statsToHerdProfileFields(getPreviewStatsArray()),
                }
              : undefined
          }
          onSubmit={(data) => {
            createMutation.mutate(data, {
              onSuccess: () => {
                setSaveOpen(false);
                setPreview(null);
              },
            });
          }}
          onCancel={() => setSaveOpen(false)}
          isLoading={createMutation.isPending}
        />
      </Modal>
    </Stack>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export function DataSourcePicker(): ReactElement {
  const { activePreset, dataset: uploadedDataset } = useUploadedCows();

  // Derive initial active source from context state
  const [activeSource, setActiveSource] = useState<SourceKey | null>(() => {
    if (activePreset) return activePreset.dataset;
    if (uploadedDataset) return "upload";
    return null;
  });

  // Keep active source in sync when context changes from outside (e.g. page reload)
  useEffect(() => {
    if (activePreset && activeSource !== activePreset.dataset) {
      setActiveSource(activePreset.dataset);
    }
  }, [activePreset, activeSource]);

  return (
    <Stack gap="md">
      <div>
        <Text size="sm" fw={500}>Data source</Text>
        <Text size="xs" c="dimmed" mt={2}>
          Pick a preset farm dataset or upload your own file to start analyzing lactation curves.
        </Text>
      </div>

      {/* Source tiles */}
      <Group gap="sm">
        {SOURCE_OPTIONS.map((opt) => {
          const isActive = activeSource === opt.value;
          return (
            <UnstyledButton
              key={opt.value}
              onClick={() => setActiveSource(opt.value)}
              style={{ flex: 1, minWidth: 140 }}
            >
              <Paper
                withBorder
                p="md"
                radius="md"
                h="100%"
                style={{
                  borderColor: isActive ? "var(--mantine-color-violet-6)" : undefined,
                  borderWidth: isActive ? 2 : 1,
                  cursor: "pointer",
                  transition: "border-color 0.12s",
                }}
              >
                <Text size="sm" fw={600}>{opt.label}</Text>
                <Text size="xs" c="dimmed" mt={2}>{opt.description}</Text>
              </Paper>
            </UnstyledButton>
          );
        })}
      </Group>

      {/* Panel for selected source */}
      {activeSource === "aurora" && <PresetPanel dataset="aurora" />}
      {activeSource === "sunnyside" && <PresetPanel dataset="sunnyside" />}
      {activeSource === "upload" && <UploadPanel />}
    </Stack>
  );
}
