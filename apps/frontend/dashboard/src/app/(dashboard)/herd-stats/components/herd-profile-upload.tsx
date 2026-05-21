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
import { HERD_STATS_METADATA } from "@/data/herd-stats-metadata";
import { statsToHerdProfileFields } from "@/lib/herd-profile-utils";
import { useUploadedCows } from "@/app/providers/uploaded-cows-provider";
import type { HerdProfileUploadResponse } from "@/types/api";
import { HerdProfileForm } from "./herd-profile-form";
import { useHerdProfileUpload } from "../hooks/use-herd-profile-upload";
import { useCreateHerdProfile } from "../hooks/use-herd-profiles";

type FormatKey = "aggregated" | "icar_test_day" | "dairycom_test_day";

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
  "1483,9/10/2019,MilkRecording,6/3/2019,2/9/2009,7,99,48.3\n" +
  "1528,6/18/2019,MilkRecording,6/5/2019,1/3/2011,5,13,45.8\n" +
  "1528,7/16/2019,MilkRecording,6/5/2019,1/3/2011,5,41,50.2\n" +
  "1528,8/13/2019,MilkRecording,6/5/2019,1/3/2011,5,69,47.0\n";

const DAIRYCOM_TEMPLATE =
  '"ID";"TestDate";"DIM";"MILK";"PCTF";"PCTP";"FCM";"305ME";"RELV";"SCC";"LS";"PEN";\n' +
  "     407 ;09/27/24; 181 ; 97  ;  3,1 ;  3,0 ; 91 ;29920 ;  97 ;   22 ;0,8 ;  6 ;\n" +
  "     407 ;10/25/24; 209 ; 95  ;  3,9 ;  3,1 ;101 ;31020 ; 101 ;   54 ;2,1 ;  6 ;\n" +
  "     407 ;11/22/24; 237 ; 93  ;  4,0 ;  3,2 ; 99 ;31960 ; 104 ;   27 ;1,1 ; 15 ;\n" +
  "     512 ;09/27/24;  42 ;110  ;  3,2 ;  3,0 ;107 ;28500 ;  95 ;   18 ;0,6 ;  6 ;\n" +
  "     512 ;10/25/24;  70 ;103  ;  3,5 ;  3,0 ;102 ;29100 ;  97 ;   21 ;0,7 ;  6 ;\n";

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
    label: "Aggregated herd stats",
    blurb:
      "One row per herd summary, with the 10 canonical columns already pre-averaged. Column order is flexible; missing columns are left at slider defaults. Use this format if your herd-management platform already exports per-herd aggregates.",
    columns: HERD_STATS_METADATA.map((m) => ({
      name: m.name,
      description: `${m.description} (${m.unit || "0–1 score"}, typical ${m.rawMin}–${m.rawMax})`,
      required: false,
    })),
    template: buildAggregatedTemplate,
    templateName: "herd_stats_template_aggregated.csv",
  },
  icar_test_day: {
    label: "Standard test-day records",
    blurb:
      "One row per cow per milk recording, as exported by milk-recording software. We aggregate across cows to derive AchievedMilk, Achieved21Milk, Achieved75Milk, Achieved305Milk (trapezoidal test-interval method) and DaysInMilk. Parity is also detected and shown as a hint for the autoencoder. All other herd stats remain at slider defaults.",
    columns: [
      { name: "TestId", description: "Unique cow identifier", required: true },
      { name: "DaysInMilk", description: "Days since calving for this record", required: true },
      { name: "DailyMilkingYield", description: "Daily milk yield in kg", required: true },
      {
        name: "Parity",
        description: "Lactation number - used to pick the dominant parity across the herd",
        required: false,
      },
      {
        name: "EventType",
        description: "Only rows with value MilkRecording are kept",
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
  dairycom_test_day: {
    label: "DairyCom export",
    blurb:
      "Semicolon-separated DairyCom (Cornell-style) export. Milk values in pounds are converted to kilograms automatically. We use the 305ME column directly as the 305-day equivalent yield; the remaining milk stats are derived from MILK at the standard DIM windows. European decimal notation (3,1) is handled.",
    columns: [
      { name: "ID", description: "Unique cow identifier", required: true },
      { name: "DIM", description: "Days in milk for this test day", required: true },
      {
        name: "MILK",
        description: "Daily milk yield in lbs (auto-converted to kg)",
        required: true,
      },
      {
        name: "305ME",
        description: "305-day mature equivalent in lbs (preferred source for Achieved305Milk)",
        required: false,
      },
      {
        name: "PCTF / PCTP / FCM / RELV / SCC / LS / PEN",
        description: "Ignored for herd-stat aggregation, accepted in the file",
        required: false,
      },
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

const FORMAT_LABELS: Record<FormatKey, string> = {
  aggregated: "Aggregated",
  icar_test_day: "Standard test-day",
  dairycom_test_day: "DairyCom",
};

export function HerdProfileUpload(): ReactElement {
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadMutation = useHerdProfileUpload();
  const createMutation = useCreateHerdProfile();
  const { setDataset } = useUploadedCows();
  const [preview, setPreview] = useState<HerdProfileUploadResponse | null>(null);
  const [saveOpen, setSaveOpen] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState<FormatKey>("icar_test_day");
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
          server-side and aggregate per-cow records when needed. The detection hint below drives the
          example template and the format docs only.
        </Text>

        <SegmentedControl
          size="xs"
          value={selectedFormat}
          onChange={(v) => setSelectedFormat(v as FormatKey)}
          data={(Object.keys(FORMATS) as FormatKey[]).map((k) => ({
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
                {preview.cow_count != null && ` · ${preview.cow_count} cows`}
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
                {preview.cows.length} cow record(s) from <Code>{uploadedFilename}</Code> saved for
                use in the{" "}
                <Link href="/curves" style={{ textDecoration: "underline" }}>
                  Curves tab
                </Link>{" "}
                - pick individual cows from the "Uploaded" group or use the random-cow button.
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
