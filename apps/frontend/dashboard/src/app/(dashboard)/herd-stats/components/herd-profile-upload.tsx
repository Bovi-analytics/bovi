"use client";

import type { ReactElement } from "react";
import { useRef, useState } from "react";
import {
  Accordion,
  Alert,
  Badge,
  Button,
  Code,
  Group,
  Modal,
  SegmentedControl,
  Stack,
  Table,
  Text,
} from "@mantine/core";
import { AlertCircle, CheckCircle2, Download } from "lucide-react";
import Link from "next/link";
import { HERD_STATS_METADATA, VISIBLE_HERD_STATS_METADATA } from "@/data/herd-stats-metadata";
import { statsToHerdProfileFields } from "@/lib/herd-profile-utils";
import { useUploadedCows } from "@/app/providers/uploaded-cows-provider";
import type { HerdProfileUploadResponse } from "@/types/api";
import { HerdProfileForm } from "./herd-profile-form";
import { useHerdProfileUpload } from "../hooks/use-herd-profile-upload";
import { useCreateHerdProfile } from "../hooks/use-herd-profiles";

type FormatKey = "aggregated" | "icar_test_day";
type SelectableFormatKey = FormatKey;

interface FormatMeta {
  label: string;
  blurb: string;
  columns: Array<{ name: string; description: string; required: boolean }>;
  template: () => string;
  templateName: string;
}

const ICAR_TEMPLATE =
  "TestId,TestDate,CalvingDate,BirthDate,Parity,DaysInMilk,DailyMilkingYield\n" +
  "1483,6/18/2019,6/3/2019,2/9/2009,7,15,49.1\n" +
  "1483,7/16/2019,6/3/2019,2/9/2009,7,43,53.4\n" +
  "1483,8/13/2019,6/3/2019,2/9/2009,7,71,52.1\n" +
  "1483,9/10/2019,6/3/2019,2/9/2009,7,99,48.3\n" +
  "1528,6/18/2019,6/5/2019,1/3/2011,5,13,45.8\n" +
  "1528,7/16/2019,6/5/2019,1/3/2011,5,41,50.2\n" +
  "1528,8/13/2019,6/5/2019,1/3/2011,5,69,47.0\n";

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
      "One row per lactation per milk recording, as exported by milk-recording software. We aggregate across lactations to derive AchievedMilk, Achieved21Milk, Achieved75Milk, Achieved305Milk (trapezoidal test-interval method) and DaysInMilk. Parity is also detected and shown as a hint for the AI autoencoder. All other herd stats remain at slider defaults.",
    columns: [
      { name: "TestId", description: "Unique lactation identifier", required: true },
      { name: "DaysInMilk", description: "Days since calving for this record", required: true },
      {
        name: "DailyMilkingYield",
        description:
          "Summed cumulative milk yield of all milkings of one day (24h milk yield).",
        required: true,
      },
      {
        name: "Parity",
        description: "Lactation number - used to pick the dominant parity across the herd",
        required: false,
      },
      {
        name: "TestDate / CalvingDate / BirthDate",
        description: "Ignored; may be present without issue",
        required: false,
      },
    ],
    template: () => ICAR_TEMPLATE,
    templateName: "herd_stats_template_icar.csv",
  },
  aggregated: {
    label: "Herd summary",
    blurb:
      "One row per herd summary, with the 10 canonical columns already pre-averaged. Column order is flexible; missing columns are left at slider defaults. Use this format if your herd-management platform already exports per-herd aggregates.",
    columns: VISIBLE_HERD_STATS_METADATA.map((m) => ({
      name: m.name,
      description: `${m.description} (${m.unit || "0–1 score"}, typical ${m.rawMin}–${m.rawMax})`,
      required: false,
    })),
    template: buildAggregatedTemplate,
    templateName: "herd_stats_template_aggregated.csv",
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

const FORMAT_LABELS: Record<FormatKey, string> = {
  aggregated: "Herd summary",
  icar_test_day: "Milk Recordings",
};

export function HerdProfileUpload(): ReactElement {
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadMutation = useHerdProfileUpload();
  const createMutation = useCreateHerdProfile();
  const { setDataset } = useUploadedCows();
  const [preview, setPreview] = useState<HerdProfileUploadResponse | null>(null);
  const [saveOpen, setSaveOpen] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState<SelectableFormatKey>("icar_test_day");
  const [uploadedFilename, setUploadedFilename] = useState<string | null>(null);

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
          response.format_detected === "icar_test_day"
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

  const activeFormat = FORMATS[selectedFormat];
  const detectedMismatch = preview !== null && preview.format_detected !== selectedFormat;

  return (
    <>
      <Stack gap="sm">
        <Text size="sm" fw={500}>
          Import from CSV
        </Text>
        <Text size="xs">
          Pick the format that matches your export, then upload the file. We auto-detect the format
          server-side and aggregate per-lactation records when needed. The detection hint below
          drives the example template and the format docs only.
        </Text>

        <SegmentedControl
          size="xs"
          value={selectedFormat}
          onChange={(v) => setSelectedFormat(v as SelectableFormatKey)}
          data={(Object.keys(FORMATS) as SelectableFormatKey[]).map((k) => ({
            value: k,
            label: FORMAT_LABELS[k],
          }))}
        />

        <Text size="xs">{activeFormat.blurb}</Text>

        <Accordion variant="contained">
          <Accordion.Item value="format">
            <Accordion.Control>
              <Text size="xs" fw={500}>
                {activeFormat.label} - expected columns
              </Text>
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
                        <Table.Td>
                          <Code>{c.name}</Code>
                        </Table.Td>
                        <Table.Td>{c.description}</Table.Td>
                        <Table.Td>
                          {c.required ? (
                            <Badge size="xs" color="red" variant="light">
                              required
                            </Badge>
                          ) : (
                            <Text size="xs">optional</Text>
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
                  Download example {FORMAT_LABELS[selectedFormat]} CSV
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
                from <Code>{FORMAT_LABELS[selectedFormat]}</Code> - preview below uses the detected
                format.
              </Alert>
            )}
            <Group gap="xs">
              <Badge variant="light">{FORMAT_LABELS[preview.format_detected]}</Badge>
              <Text size="xs">
                {preview.row_count.toLocaleString()} row(s) processed
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
                {preview.cows.length} lactation record(s) from <Code>{uploadedFilename}</Code>{" "}
                saved. Continue to the{" "}
                <Link href="/herd-profiles" style={{ textDecoration: "underline" }}>
                  Herd Profiles tab
                </Link>{" "}
                before opening Curves.
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
              <Button size="sm" color="violet" onClick={() => setSaveOpen(true)}>
                Save as profile…
              </Button>
              <Button size="sm" variant="subtle" onClick={() => setPreview(null)}>
                Discard
              </Button>
            </Group>
          </Stack>
        )}
      </Stack>

      <Modal
        opened={saveOpen}
        onClose={() => setSaveOpen(false)}
        title="Save uploaded herd profile"
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
    </>
  );
}
