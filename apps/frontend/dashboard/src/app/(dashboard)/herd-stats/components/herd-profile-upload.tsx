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
  Stack,
  Table,
  Text,
} from "@mantine/core";
import { AlertCircle, Download } from "lucide-react";
import { HERD_STATS_METADATA } from "@/data/herd-stats-metadata";
import { statsToHerdProfileFields } from "@/lib/herd-profile-utils";
import type { HerdProfileUploadResponse } from "@/types/api";
import { HerdProfileForm } from "./herd-profile-form";
import { useHerdProfileUpload } from "../hooks/use-herd-profile-upload";
import { useCreateHerdProfile } from "../hooks/use-herd-profiles";

function downloadCsvTemplate(): void {
  const headers = HERD_STATS_METADATA.map((m) => m.name).join(",");
  // Example row: midpoint of each raw range
  const exampleRow = HERD_STATS_METADATA.map((m) => {
    const mid = (m.rawMin + m.rawMax) / 2;
    return Math.round(mid * 100) / 100;
  }).join(",");
  const csv = `${headers}\n${exampleRow}\n`;
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "herd_stats_template.csv";
  a.click();
  URL.revokeObjectURL(url);
}

export function HerdProfileUpload(): ReactElement {
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadMutation = useHerdProfileUpload();
  const createMutation = useCreateHerdProfile();
  const [preview, setPreview] = useState<HerdProfileUploadResponse | null>(null);
  const [saveOpen, setSaveOpen] = useState(false);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    uploadMutation.mutate(file, { onSuccess: setPreview });
    e.target.value = "";
  }

  function getPreviewStatsArray(): number[] {
    if (!preview) return [];
    return HERD_STATS_METADATA.map((meta) => preview.stats[meta.name] ?? 0);
  }

  return (
    <>
      <Stack gap="sm">
        <Text size="sm" fw={500}>
          Import from CSV
        </Text>
        <Text size="xs" c="dimmed">
          Upload a CSV file with herd statistics in their natural units (kg, days). The system
          normalises values automatically. Each row represents one cow or one time period — multiple
          rows are averaged before saving. A single-row file is treated as an already-aggregated
          herd summary.
        </Text>

        <Accordion variant="contained">
          <Accordion.Item value="format">
            <Accordion.Control>
              <Text size="xs" fw={500}>
                Expected CSV format
              </Text>
            </Accordion.Control>
            <Accordion.Panel>
              <Stack gap="xs">
                <Text size="xs" c="dimmed">
                  The CSV must contain a header row with the exact column names below. Column order
                  does not matter. Missing columns will be filled with defaults.
                </Text>
                <Table striped withColumnBorders fz="xs">
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Column name</Table.Th>
                      <Table.Th>Description</Table.Th>
                      <Table.Th>Unit</Table.Th>
                      <Table.Th>Typical range</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {HERD_STATS_METADATA.map((meta) => (
                      <Table.Tr key={meta.name}>
                        <Table.Td>
                          <Code>{meta.name}</Code>
                        </Table.Td>
                        <Table.Td>{meta.description}</Table.Td>
                        <Table.Td>
                          {meta.unit ? (
                            <Badge size="xs" variant="light">
                              {meta.unit}
                            </Badge>
                          ) : (
                            <Text size="xs" c="dimmed">
                              0–1 score
                            </Text>
                          )}
                        </Table.Td>
                        <Table.Td>
                          {meta.rawMin} – {meta.rawMax}
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
                  onClick={downloadCsvTemplate}
                >
                  Download CSV template
                </Button>
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>

        <input
          ref={inputRef}
          type="file"
          accept=".csv"
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
            <Text size="xs" c="dimmed">
              Format: {preview.format_detected} · {preview.row_count} row(s) processed
            </Text>
            {preview.warnings.map((w, i) => (
              <Alert key={i} icon={<AlertCircle size={14} />} color="yellow">
                {w}
              </Alert>
            ))}
            <Table striped withColumnBorders fz="xs">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Stat</Table.Th>
                  <Table.Th>Normalised (0–1)</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {HERD_STATS_METADATA.map((meta) => (
                  <Table.Tr key={meta.name}>
                    <Table.Td>{meta.label}</Table.Td>
                    <Table.Td>{preview.stats[meta.name]?.toFixed(3) ?? "—"}</Table.Td>
                  </Table.Tr>
                ))}
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
