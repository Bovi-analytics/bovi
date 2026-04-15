"use client";

import type { ReactElement } from "react";
import { useRef, useState } from "react";
import { Alert, Button, Group, Modal, Stack, Table, Text } from "@mantine/core";
import { AlertCircle } from "lucide-react";
import { HERD_STATS_METADATA } from "@/data/herd-stats-metadata";
import { statsToHerdProfileFields } from "@/lib/herd-profile-utils";
import type { HerdProfileUploadResponse } from "@/types/api";
import { HerdProfileForm } from "./herd-profile-form";
import { useHerdProfileUpload } from "../hooks/use-herd-profile-upload";
import { useCreateHerdProfile } from "../hooks/use-herd-profiles";

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
    e.target.value = ""; // allow re-upload of same file
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
          Upload a CSV with herd stat columns. One-row CSVs are treated as aggregated; multi-row
          CSVs are averaged per column. Values are normalized to 0–1 automatically.
        </Text>

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
            <Table striped withColumnBorders>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Stat</Table.Th>
                  <Table.Th>Normalized (0–1)</Table.Th>
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
