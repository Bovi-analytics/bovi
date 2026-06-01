"use client";

import { Fragment, useEffect, useRef, useState } from "react";
import type { ReactElement } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Accordion,
  ActionIcon,
  Alert,
  Badge,
  Button,
  Code,
  Group,
  Loader,
  Modal,
  Paper,
  SegmentedControl,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
  Tooltip,
  UnstyledButton,
} from "@mantine/core";
import { AlertCircle, CheckCircle2, ChevronRight, Download, Info, Trash2 } from "lucide-react";
import Link from "next/link";
import { VISIBLE_HERD_STATS_METADATA } from "@/data/herd-stats-metadata";
import { useUploadedCows, type UploadedDataset } from "@/app/providers/uploaded-cows-provider";
import { ActiveDatasetPanel } from "@/components/dashboard/active-dataset-panel";
import { CenteredLoader } from "@/components/dashboard/centered-loader";
import { usePresetCounts } from "@/app/(dashboard)/curves/hooks/use-preset-counts";
import { usePresetDataset } from "@/app/(dashboard)/curves/hooks/use-preset-dataset";
import {
  deleteUploadedDataset,
  getUploadedDataset,
  listOrganizationMembers,
  listUploadedDatasets,
} from "@/lib/api-client";
import { useAuth } from "@/lib/auth";
import { UPLOAD_LIMIT_DESCRIPTION } from "@/lib/upload-limits";
import { getInitialDataSource, type DataSourcePickerSourceKey } from "./data-source-picker-state";
import type {
  HerdProfileUploadResponse,
  PresetDatasetKey,
  PresetPeriodKey,
  PresetSizeKey,
  UploadedDatasetDetail,
} from "@/types/api";
import { useHerdProfileUpload } from "../hooks/use-herd-profile-upload";

/* ------------------------------------------------------------------ */
/*  Types & constants                                                  */
/* ------------------------------------------------------------------ */

type SourceKey = PresetDatasetKey | "upload" | "saved";
type FormatKey = "aggregated" | "icar_test_day";
type SelectableFormatKey = FormatKey;
type TestDayMappingKey = "cow_id" | "dim" | "milk_kg" | "parity" | "herd_id" | "event_type";

interface SourceOption<T extends SourceKey = SourceKey> {
  value: T;
  label: string;
  description: string;
}

const PRESET_SOURCE_OPTIONS: Array<SourceOption<PresetDatasetKey>> = [
  {
    value: "aurora",
    label: "Demo herd A",
    description: "Anonymized herd · 2023-2025",
  },
  {
    value: "sunnyside",
    label: "Demo herd B",
    description: "Anonymized herd · 2000-2026",
  },
];

const UPLOAD_SOURCE_OPTION: SourceOption<"upload"> = {
  value: "upload",
  label: "Upload a file",
  description: "Use your own CSV",
};

const SAVED_SOURCE_OPTION: SourceOption = {
  value: "saved",
  label: "Saved datasets",
  description: "Reuse your own uploaded datasets",
};

const OTHER_SOURCE_OPTIONS: SourceOption[] = [UPLOAD_SOURCE_OPTION, SAVED_SOURCE_OPTION];

const SIZE_OPTIONS = [
  { value: "small", label: "Small" },
  { value: "medium", label: "Medium" },
  { value: "large", label: "Large" },
] satisfies Array<{ value: PresetSizeKey; label: string }>;

const PERIOD_OPTIONS = [
  { value: "recent", label: "Recent" },
  { value: "old", label: "Old" },
  { value: "mixed", label: "Mixed" },
];

function sizeOptionsWithCounts(
  counts: Record<string, Record<string, number>> | undefined,
  period: PresetPeriodKey
): Array<{ value: PresetSizeKey; label: string }> {
  return SIZE_OPTIONS.map((option) => {
    const count = counts?.[period]?.[option.value];
    return {
      ...option,
      label: count === undefined ? option.label : `${option.label} (${count.toLocaleString()})`,
    };
  });
}

interface FormatMeta {
  label: string;
  blurb: string;
  columns: Array<{ name: string; description: string; required: boolean; help?: string }>;
  template: () => string;
  templateName: string;
}

const ICAR_TEMPLATE =
  "TestId,TestDate,CalvingDate,BirthDate,Parity,DaysInMilk,DailyMilkingYield\n" +
  "1483,6/18/2019,6/3/2019,2/9/2009,7,15,49.1\n" +
  "1483,7/16/2019,6/3/2019,2/9/2009,7,43,53.4\n" +
  "1483,8/13/2019,6/3/2019,2/9/2009,7,71,52.1\n" +
  "1528,6/18/2019,6/5/2019,1/3/2011,5,13,45.8\n";

function buildAggregatedTemplate(): string {
  const headers = VISIBLE_HERD_STATS_METADATA.map((m) => m.name).join(",");
  const exampleRow = VISIBLE_HERD_STATS_METADATA.map((m) => {
    const mid = (m.rawMin + m.rawMax) / 2;
    return Math.round(mid * 100) / 100;
  }).join(",");
  return `${headers}\n${exampleRow}\n`;
}

const FORMATS: Record<SelectableFormatKey, FormatMeta> = {
  icar_test_day: {
    label: "Milk Recordings",
    blurb:
      "One row per lactation per recording date - the raw export you get from milk recording software. We calculate herd averages from these records automatically. Also extracts individual lactation data for use on the Curves tab.",
    columns: [
      {
        name: "TestId",
        description: "Unique lactation identifier",
        required: true,
        help: "The TestId is an unique identifier for a lactation, which can be used to group records belonging to the same lactation together. It is not the same as a cow ID, as a cow can have multiple lactations (e.g., across different calvings).",
      },
      { name: "DaysInMilk", description: "Days since calving", required: true },
      {
        name: "DailyMilkingYield",
        description: "Summed cumulative milk yield of all milkings of one day (24h milk yield).",
        required: true,
      },
      { name: "Parity", description: "Lactation number", required: false },
    ],
    template: () => ICAR_TEMPLATE,
    templateName: "herd_stats_template_icar.csv",
  },
  aggregated: {
    label: "Herd summary",
    blurb:
      "A single row with the herd statistics already averaged across your herd. Use this when your farm software exports per-herd aggregates.",
    columns: VISIBLE_HERD_STATS_METADATA.map((m) => ({
      name: m.name,
      description: `${m.description} (${m.unit || "0–1 score"}, package range ${m.rawMin}–${m.rawMax})`,
      required: false,
    })),
    template: buildAggregatedTemplate,
    templateName: "herd_stats_template_aggregated.csv",
  },
};

const REQUIRED_MAPPING_KEYS: TestDayMappingKey[] = ["cow_id", "dim", "milk_kg"];
const OPTIONAL_MAPPING_KEYS: TestDayMappingKey[] = ["parity", "herd_id", "event_type"];
const MAPPING_LABELS: Record<TestDayMappingKey, string> = {
  cow_id: "Lactation ID",
  dim: "Days in milk",
  milk_kg: "Milk yield",
  parity: "Parity",
  herd_id: "Herd ID",
  event_type: "Event type",
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

function toUploadedDataset(
  response: HerdProfileUploadResponse,
  filename: string,
  datasetName: string,
  uploadedAt = new Date().toISOString()
): UploadedDataset {
  const name = datasetName.trim() || cleanDatasetName(filename);
  return {
    id: response.upload_id ?? `${Date.now()}-${name}`,
    name,
    format: "icar_test_day",
    uploadedAt,
    rowCount: response.row_count,
    cowCount: response.cow_count ?? response.cows.length,
    detectedParity: response.detected_parity ?? null,
    columns: response.columns,
    columnMapping: response.column_mapping,
    stats: response.stats,
    rawStats: response.raw_stats,
    cows: response.cows.map((c) => ({
      cowId: c.cow_id,
      parity: c.parity,
      dim: c.dim,
      milkrecordings: c.milk_kg,
    })),
  };
}

function toUploadedDatasetFromDetail(detail: UploadedDatasetDetail): UploadedDataset {
  return {
    id: detail.id,
    name: detail.name,
    format: "icar_test_day",
    uploadedAt: detail.uploaded_at ?? new Date().toISOString(),
    userId: detail.user_id,
    userName: detail.user_name,
    userEmail: detail.user_email,
    organizationId: detail.organization_id,
    organizationName: detail.organization_name,
    rowCount: detail.row_count,
    cowCount: detail.cow_count ?? detail.cows.length,
    detectedParity: detail.detected_parity ?? null,
    columns: detail.columns,
    columnMapping: detail.column_mapping,
    stats: detail.stats,
    rawStats: detail.raw_stats,
    cows: detail.cows.map((c) => ({
      cowId: c.cow_id,
      parity: c.parity,
      dim: c.dim,
      milkrecordings: c.milk_kg,
    })),
  };
}

function ownerLabel(item: {
  user_name?: string | null;
  user_email?: string | null;
  user_id?: number | null;
}): string {
  return item.user_name || item.user_email || (item.user_id ? `User #${item.user_id}` : "-");
}

function mappingSummary(mapping: Readonly<Record<string, string>> | undefined): string {
  if (!mapping) return "-";
  return REQUIRED_MAPPING_KEYS.map((key) => `${MAPPING_LABELS[key]}: ${mapping[key] ?? "-"}`).join(
    " · "
  );
}

function cleanDatasetName(filename: string): string {
  return (
    filename
      .replace(/\.[^.]+$/, "")
      .replace(/[_-]+/g, " ")
      .trim() || filename
  );
}

function formatUploadDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
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
  const {
    data: presetData,
    isLoading,
    isError,
  } = usePresetDataset(dataset, selectedSize, selectedPeriod);
  const { data: presetCounts } = usePresetCounts(dataset);

  const isActive =
    activePreset?.dataset === dataset &&
    activePreset.size === selectedSize &&
    activePreset.period === selectedPeriod;

  function activate() {
    setActivePreset({ dataset, size: selectedSize, period: selectedPeriod });
  }

  return (
    <Stack gap="lg">
      <Stack gap={6}>
        <Text size="sm" fw={600}>
          Sample size
        </Text>
        <SegmentedControl
          size="sm"
          value={selectedSize}
          onChange={(v) => setSelectedSize(v as PresetSizeKey)}
          data={sizeOptionsWithCounts(presetCounts?.counts, selectedPeriod)}
        />
      </Stack>
      <Stack gap={6}>
        <Text size="sm" fw={600}>
          Time period
        </Text>
        <SegmentedControl
          size="sm"
          value={selectedPeriod}
          onChange={(v) => setSelectedPeriod(v as PresetPeriodKey)}
          data={PERIOD_OPTIONS}
        />
      </Stack>

      <Group gap="sm" align="center">
        {isLoading && <Loader size="sm" />}
        {presetData && !isLoading && (
          <Badge color="violet" variant="light" size="lg">
            {presetData.cow_count.toLocaleString()} lactations
          </Badge>
        )}
        {!isActive && (
          <Button size="md" color="violet" onClick={activate} disabled={isLoading || !presetData}>
            Use this demo herd
          </Button>
        )}
      </Group>

      {isActive && (
        <Alert color="violet" variant="light">
          <Group justify="space-between" align="center">
            <Text size="sm">
              Demo herd active - {presetData?.cow_count.toLocaleString()} lactations ready.
            </Text>
            <Button
              component={Link}
              href="/herd-profiles"
              size="sm"
              color="violet"
              rightSection={<ChevronRight size={14} />}
            >
              Go to Herd Profiles
            </Button>
          </Group>
        </Alert>
      )}

      {isError && (
        <Alert icon={<AlertCircle size={16} />} color="red">
          Demo herd unavailable - make sure CONNECTION_STRING is configured and the preprocessing
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
  const [selectedFormat, setSelectedFormat] = useState<SelectableFormatKey>("icar_test_day");
  const [preview, setPreview] = useState<HerdProfileUploadResponse | null>(null);
  const [uploadedFilename, setUploadedFilename] = useState<string | null>(null);
  const [datasetName, setDatasetName] = useState("");
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [mappingOpen, setMappingOpen] = useState(false);
  const [mappingDraft, setMappingDraft] = useState<Record<TestDayMappingKey, string>>({
    cow_id: "",
    dim: "",
    milk_kg: "",
    parity: "",
    herd_id: "",
    event_type: "",
  });
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const uploadMutation = useHerdProfileUpload();
  const { dataset: uploadedDataset, saveDataset } = useUploadedCows();

  const activeFormat = FORMATS[selectedFormat];
  const detectedMismatch = preview !== null && preview.format_detected !== selectedFormat;

  function activateResponse(response: HerdProfileUploadResponse, filename: string) {
    setPreview(response);
    setUploadedFilename(filename);
    if (response.cows.length > 0 && response.format_detected === "icar_test_day") {
      saveDataset(toUploadedDataset(response, filename, datasetName));
      void queryClient.invalidateQueries({ queryKey: ["uploaded-datasets"] });
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const filename = file.name;
    setPendingFile(file);
    setDatasetName(cleanDatasetName(filename));
    uploadMutation.mutate(
      { file },
      {
        onSuccess: (response) => {
          setUploadedFilename(filename);
          setPreview(response);
          if (response.mapping_required) {
            setMappingDraft({
              cow_id: response.column_mapping.cow_id ?? "",
              dim: response.column_mapping.dim ?? "",
              milk_kg: response.column_mapping.milk_kg ?? "",
              parity: response.column_mapping.parity ?? "",
              herd_id: response.column_mapping.herd_id ?? "",
              event_type: response.column_mapping.event_type ?? "",
            });
            setMappingOpen(true);
            return;
          }
          activateResponse(response, filename);
        },
      }
    );
    e.target.value = "";
  }

  function confirmMapping() {
    if (!pendingFile || !uploadedFilename) return;
    const cleaned = Object.fromEntries(
      Object.entries(mappingDraft).filter(([, value]) => value.trim())
    );
    uploadMutation.mutate(
      { file: pendingFile, columnMapping: cleaned },
      {
        onSuccess: (response) => {
          setMappingOpen(false);
          activateResponse(response, uploadedFilename);
        },
      }
    );
  }

  const FORMAT_LABELS: Record<FormatKey, string> = {
    aggregated: FORMATS.aggregated.label,
    icar_test_day: FORMATS.icar_test_day.label,
  };

  return (
    <Stack gap="lg">
      <Stack gap={6}>
        <Text size="sm" fw={600}>
          File format
        </Text>
        <SegmentedControl
          size="sm"
          value={selectedFormat}
          onChange={(v) => setSelectedFormat(v as SelectableFormatKey)}
          data={(Object.keys(FORMATS) as SelectableFormatKey[]).map((k) => ({
            value: k,
            label: FORMATS[k].label,
          }))}
        />
        <Text size="sm" mt={4}>
          {activeFormat.blurb}
        </Text>
      </Stack>

      <Accordion variant="contained">
        <Accordion.Item value="columns">
          <Accordion.Control>
            <Text size="sm" fw={600}>
              Expected columns
            </Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Stack gap="sm">
              <Table striped withColumnBorders fz="sm">
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
                      <Table.Td>
                        <Group gap={4} wrap="nowrap">
                          <Code>{c.name}</Code>
                          {c.help && (
                            <Tooltip label={c.help} multiline w={320}>
                              <Info size={14} className="text-muted-foreground" />
                            </Tooltip>
                          )}
                        </Group>
                      </Table.Td>
                      <Table.Td>{c.description}</Table.Td>
                      <Table.Td>
                        {c.required ? (
                          <Badge size="sm" color="red" variant="light">
                            required
                          </Badge>
                        ) : (
                          <Text size="sm">optional</Text>
                        )}
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
              <Button
                size="sm"
                variant="light"
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
        size="md"
        onClick={() => inputRef.current?.click()}
        loading={uploadMutation.isPending}
        style={{ width: "fit-content" }}
      >
        Choose CSV file…
      </Button>
      <Text size="xs" c="dimmed">
        {UPLOAD_LIMIT_DESCRIPTION}
      </Text>

      {uploadMutation.isError && (
        <Alert icon={<AlertCircle size={16} />} color="red" title="Upload failed">
          {(uploadMutation.error as Error).message}
        </Alert>
      )}

      {preview && (
        <Stack gap="sm">
          {preview.mapping_required && (
            <Alert icon={<AlertCircle size={14} />} color="blue">
              Review the column mapping before this dataset is selected.
            </Alert>
          )}
          {detectedMismatch && (
            <Alert icon={<AlertCircle size={14} />} color="blue">
              Detected format: <Code>{FORMAT_LABELS[preview.format_detected]}</Code>. Switched from{" "}
              <Code>{FORMAT_LABELS[selectedFormat]}</Code>.
            </Alert>
          )}
          <Group gap="xs">
            <Badge variant="light">{FORMAT_LABELS[preview.format_detected]}</Badge>
            <Text size="xs">
              {preview.row_count.toLocaleString()} row(s)
              {preview.cow_count != null && ` · ${preview.cow_count} lactations`}
              {preview.detected_parity != null && ` · dominant parity ${preview.detected_parity}`}
            </Text>
          </Group>
          {preview.warnings.map((w, i) => (
            <Alert key={i} icon={<AlertCircle size={14} />} color="yellow">
              {w}
            </Alert>
          ))}
          {preview.cows.length > 0 && uploadedFilename && (
            <Alert icon={<CheckCircle2 size={14} />} color="green">
              {preview.cows.length} lactation records from <Code>{uploadedFilename}</Code>{" "}
              {uploadedDataset?.name === uploadedFilename
                ? "selected as the active dataset."
                : "ready."}
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
              {VISIBLE_HERD_STATS_METADATA.map((meta) => {
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
                        <Text size="xs">-</Text>
                      )}
                    </Table.Td>
                    <Table.Td>
                      {filled ? (
                        preview.stats[meta.name]?.toFixed(3)
                      ) : (
                        <Text size="xs">slider default</Text>
                      )}
                    </Table.Td>
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
          <Group>
            {preview.mapping_required && (
              <Button size="sm" color="violet" onClick={() => setMappingOpen(true)}>
                Review column mapping
              </Button>
            )}
            {uploadedDataset?.name === uploadedFilename && (
              <Button
                component={Link}
                href="/herd-profiles"
                size="sm"
                color="violet"
                rightSection={<ChevronRight size={14} />}
              >
                Go to Herd Profiles
              </Button>
            )}
            <Button size="sm" variant="subtle" onClick={() => setPreview(null)}>
              Discard
            </Button>
          </Group>
        </Stack>
      )}

      <Modal
        opened={mappingOpen}
        onClose={() => setMappingOpen(false)}
        title="Confirm column mapping"
        size="lg"
      >
        <Stack gap="md">
          <TextInput
            label="Dataset name"
            value={datasetName}
            onChange={(event) => setDatasetName(event.currentTarget.value)}
            maxLength={80}
            required
          />
          <Text size="sm">
            Match the uploaded columns to the fields needed for milk-recording datasets.
          </Text>
          {[...REQUIRED_MAPPING_KEYS, ...OPTIONAL_MAPPING_KEYS].map((key) => (
            <Select
              key={key}
              label={MAPPING_LABELS[key]}
              required={REQUIRED_MAPPING_KEYS.includes(key)}
              clearable={!REQUIRED_MAPPING_KEYS.includes(key)}
              value={mappingDraft[key] || null}
              data={(preview?.columns ?? []).map((column) => ({ value: column, label: column }))}
              onChange={(value) =>
                setMappingDraft((current) => ({ ...current, [key]: value ?? "" }))
              }
            />
          ))}
          <Group justify="flex-end">
            <Button variant="subtle" onClick={() => setMappingOpen(false)}>
              Cancel
            </Button>
            <Button
              color="violet"
              onClick={confirmMapping}
              loading={uploadMutation.isPending}
              disabled={REQUIRED_MAPPING_KEYS.some((key) => !mappingDraft[key])}
            >
              Accept mapping
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
}

function SavedDatasetsPanel(): ReactElement {
  const { selectedOrganizationId } = useAuth();
  const { dataset: uploadedDataset, setDataset, clearDataset } = useUploadedCows();
  const queryClient = useQueryClient();
  const [expandedMappingId, setExpandedMappingId] = useState<string | null>(null);
  const [scope, setScope] = useState<"mine" | "organization">("mine");
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);
  const [sort, setSort] = useState<"uploaded_at" | "name" | "user">("uploaded_at");
  const [direction, setDirection] = useState<"asc" | "desc">("desc");
  const [q, setQ] = useState("");
  const organizationId = selectedOrganizationId ?? 0;
  const userId = scope === "organization" && selectedUserId ? Number(selectedUserId) : undefined;
  const datasetOptions = {
    scope: scope === "mine" ? "mine" : "organization",
    user_id: userId,
    sort,
    direction,
    q: q.trim() || undefined,
  } as const;
  const datasetsQuery = useQuery({
    queryKey: ["uploaded-datasets", organizationId, datasetOptions],
    queryFn: () => listUploadedDatasets(organizationId, datasetOptions),
    enabled: selectedOrganizationId !== null,
  });
  const membersQuery = useQuery({
    queryKey: ["organization-members", selectedOrganizationId],
    queryFn: () => listOrganizationMembers(selectedOrganizationId as number),
    enabled: typeof selectedOrganizationId === "number",
  });
  const selectMutation = useMutation({
    mutationFn: getUploadedDataset,
    onSuccess: (detail) => setDataset(toUploadedDatasetFromDetail(detail)),
  });
  const deleteMutation = useMutation({
    mutationFn: deleteUploadedDataset,
    onSuccess: (_void, deletedId) => {
      if (uploadedDataset?.id === deletedId) {
        clearDataset();
      }
      void queryClient.invalidateQueries({ queryKey: ["uploaded-datasets"] });
    },
  });
  const savedDatasets = datasetsQuery.data ?? [];

  if (datasetsQuery.isLoading) {
    return <CenteredLoader label="Loading saved datasets..." />;
  }

  if (datasetsQuery.isError) {
    return (
      <Alert icon={<AlertCircle size={16} />} color="red">
        Failed to load saved datasets.
      </Alert>
    );
  }

  return (
    <Stack gap="md">
      <Group gap="sm" align="flex-end">
        <SegmentedControl
          size="xs"
          value={scope}
          onChange={(value) => {
            setScope(value as "mine" | "organization");
            setSelectedUserId(null);
          }}
          data={[
            { label: "My items", value: "mine" },
            { label: "Organization", value: "organization" },
          ]}
        />
        {scope === "organization" && (
          <Select
            aria-label="Filter by organization member"
            label="Person"
            size="xs"
            value={selectedUserId}
            onChange={setSelectedUserId}
            placeholder="All organization members"
            clearable
            searchable
            data={(membersQuery.data ?? []).map((member) => ({
              value: String(member.user_id),
              label: member.name || member.email || `User #${member.user_id}`,
            }))}
          />
        )}
        <TextInput
          aria-label="Search saved datasets"
          label="Search"
          placeholder="Search by name"
          value={q}
          onChange={(event) => setQ(event.currentTarget.value)}
          size="xs"
        />
        <Select
          aria-label="Sort saved datasets"
          label="Sort by"
          size="xs"
          value={sort}
          onChange={(value) => setSort((value as "uploaded_at" | "name" | "user") ?? "uploaded_at")}
          data={[
            { label: "Uploaded date", value: "uploaded_at" },
            { label: "Name", value: "name" },
            { label: "User", value: "user" },
          ]}
        />
        <Select
          aria-label="Sort direction"
          label="Order"
          size="xs"
          value={direction}
          onChange={(value) => setDirection((value as "asc" | "desc") ?? "desc")}
          data={[
            { label: "Newest first", value: "desc" },
            { label: "Oldest first", value: "asc" },
          ]}
        />
      </Group>

      {savedDatasets.length === 0 && (
        <Alert color="gray" variant="light">
          <Text size="sm">No saved datasets match this filter.</Text>
        </Alert>
      )}

      <Table striped withColumnBorders fz="sm">
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Status</Table.Th>
            <Table.Th>Dataset</Table.Th>
            <Table.Th>Uploaded by</Table.Th>
            <Table.Th>Uploaded</Table.Th>
            <Table.Th>Rows</Table.Th>
            <Table.Th>Lactations</Table.Th>
            <Table.Th>Mapping</Table.Th>
            <Table.Th>Columns</Table.Th>
            <Table.Th>Action</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {savedDatasets.map((saved) => {
            const active = uploadedDataset?.id === saved.id;
            return (
              <Fragment key={saved.id}>
                <Table.Tr>
                  <Table.Td>
                    {active ? (
                      <Badge color="violet" variant="filled">
                        Active
                      </Badge>
                    ) : (
                      <Badge color="gray" variant="light">
                        Saved
                      </Badge>
                    )}
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" fw={600} maw={180} lineClamp={1} title={saved.name}>
                      {saved.name}
                    </Text>
                    {saved.detected_parity != null && (
                      <Text size="xs" c="dimmed">
                        Dominant parity {saved.detected_parity}
                      </Text>
                    )}
                  </Table.Td>
                  <Table.Td>{ownerLabel(saved)}</Table.Td>
                  <Table.Td>
                    {saved.uploaded_at ? formatUploadDate(saved.uploaded_at) : "-"}
                  </Table.Td>
                  <Table.Td>{saved.row_count?.toLocaleString() ?? "-"}</Table.Td>
                  <Table.Td>{saved.cow_count?.toLocaleString() ?? "-"}</Table.Td>
                  <Table.Td>
                    <ActionIcon
                      aria-label="Show column mapping"
                      color="pink"
                      radius="xl"
                      variant="filled"
                      onClick={() =>
                        setExpandedMappingId(expandedMappingId === saved.id ? null : saved.id)
                      }
                    >
                      <Info size={14} />
                    </ActionIcon>
                  </Table.Td>
                  <Table.Td>
                    <Text size="xs" maw={260} lineClamp={2} title={saved.columns?.join(", ")}>
                      {saved.columns?.join(", ") ?? "-"}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Group gap="xs" wrap="nowrap">
                      <Button
                        size="xs"
                        variant={active ? "light" : "filled"}
                        color="violet"
                        disabled={active}
                        loading={selectMutation.isPending && selectMutation.variables === saved.id}
                        onClick={() => selectMutation.mutate(saved.id)}
                      >
                        {active ? "Selected" : "Select"}
                      </Button>
                      <ActionIcon
                        aria-label="Delete saved dataset"
                        color="red"
                        variant="subtle"
                        loading={deleteMutation.isPending && deleteMutation.variables === saved.id}
                        onClick={() => deleteMutation.mutate(saved.id)}
                      >
                        <Trash2 size={14} />
                      </ActionIcon>
                    </Group>
                  </Table.Td>
                </Table.Tr>
                {expandedMappingId === saved.id && (
                  <Table.Tr>
                    <Table.Td colSpan={9}>
                      <Text size="xs" c="dimmed">
                        {mappingSummary(saved.column_mapping)}
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                )}
              </Fragment>
            );
          })}
        </Table.Tbody>
      </Table>

      {uploadedDataset && (
        <Alert color="violet" variant="light">
          <Text size="sm">
            {uploadedDataset.name} active -{" "}
            {(uploadedDataset.cowCount ?? uploadedDataset.cows.length).toLocaleString()} lactations
            ready.
          </Text>
        </Alert>
      )}
    </Stack>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export function DataSourcePicker(): ReactElement {
  const { activePreset, dataset: uploadedDataset } = useUploadedCows();
  const activePresetDataset = activePreset?.dataset;

  // Open the preset section by default without selecting an active dataset.
  const [activeSource, setActiveSource] = useState<DataSourcePickerSourceKey>(() =>
    getInitialDataSource(activePreset, uploadedDataset)
  );

  // Keep active source in sync when the active preset changes from outside
  // (e.g. set on another page). Only depends on activePreset.dataset so a local
  // tile click is not immediately undone - the user can preview a different
  // preset before activating it.
  useEffect(() => {
    if (activePresetDataset) {
      setActiveSource(activePresetDataset);
    }
  }, [activePresetDataset]);

  return (
    <Stack gap="md">
      <div>
        <Text size="md" fw={700}>
          Data source
        </Text>
        <Text size="sm" mt={4}>
          Pick a built-in demo herd or upload your own file to start analyzing milk production data.
        </Text>
      </div>

      <ActiveDatasetPanel actionHref="/herd-profiles" actionLabel="Go to Herd Profiles" />

      <Stack gap="sm">
        <Text size="sm" fw={700}>
          Preset Datasets
        </Text>
        <Group gap="sm">
          {PRESET_SOURCE_OPTIONS.map((opt) => {
            const isActive = activeSource === opt.value;
            return (
              <UnstyledButton
                key={opt.value}
                onClick={() => setActiveSource(opt.value)}
                style={{ flex: 1, minWidth: 140, alignSelf: "stretch" }}
              >
                <Paper
                  withBorder
                  p="md"
                  radius="md"
                  h={104}
                  style={{
                    borderColor: isActive ? "var(--mantine-color-violet-6)" : undefined,
                    borderWidth: isActive ? 2 : 1,
                    cursor: "pointer",
                    transition: "border-color 0.12s",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                  }}
                >
                  <Text size="md" fw={700}>
                    {opt.label}
                  </Text>
                  <Text size="sm" mt={4}>
                    {opt.description}
                  </Text>
                </Paper>
              </UnstyledButton>
            );
          })}
        </Group>
      </Stack>

      {activeSource === "aurora" && <PresetPanel dataset="aurora" />}
      {activeSource === "sunnyside" && <PresetPanel dataset="sunnyside" />}

      <Stack gap="sm">
        <Text size="sm" fw={700}>
          Other sources
        </Text>
        <Group gap="sm">
          {OTHER_SOURCE_OPTIONS.map((opt) => {
            const isActive = activeSource === opt.value;
            return (
              <UnstyledButton
                key={opt.value}
                onClick={() => setActiveSource(opt.value)}
                style={{ flex: 1, minWidth: 140, alignSelf: "stretch" }}
              >
                <Paper
                  withBorder
                  p="md"
                  radius="md"
                  h={104}
                  style={{
                    borderColor: isActive ? "var(--mantine-color-violet-6)" : undefined,
                    borderWidth: isActive ? 2 : 1,
                    cursor: "pointer",
                    transition: "border-color 0.12s",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                  }}
                >
                  <Text size="md" fw={700}>
                    {opt.label}
                  </Text>
                  <Text size="sm" mt={4}>
                    {opt.description}
                  </Text>
                </Paper>
              </UnstyledButton>
            );
          })}
        </Group>
      </Stack>

      {activeSource === "upload" && <UploadPanel />}
      {activeSource === "saved" && <SavedDatasetsPanel />}
    </Stack>
  );
}
