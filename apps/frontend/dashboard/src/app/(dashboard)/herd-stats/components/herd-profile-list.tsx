"use client";

import type { ReactElement } from "react";
import { useState } from "react";
import Link from "next/link";
import { ActionIcon, Alert, Button, Group, Modal, Stack, Table, Text } from "@mantine/core";
import { CheckCircle2, ChevronRight, Pencil, Trash2 } from "lucide-react";
import { HerdProfileForm } from "./herd-profile-form";
import {
  useCreateHerdProfile,
  useDeleteHerdProfile,
  useHerdProfiles,
  useUpdateHerdProfile,
} from "../hooks/use-herd-profiles";
import type { HerdProfile, HerdProfileCreate } from "@/types/api";
import { useUploadedCows } from "@/app/providers/uploaded-cows-provider";
import { DEFAULT_HERD_STATS } from "@/data/herd-stats-metadata";
import { usePresetHerdStats } from "@/app/(dashboard)/curves/hooks/use-preset-herd-stats";

const PRESET_LABELS: Record<string, string> = {
  aurora: "Demo herd A",
  sunnyside: "Demo herd B",
};
const PERIOD_LABELS: Record<string, string> = { recent: "Recent", old: "Old", mixed: "Mixed" };
const SIZE_LABELS: Record<string, string> = { small: "Small", medium: "Medium", large: "Large" };

export function HerdProfileList(): ReactElement {
  const { data: profiles = [], isLoading } = useHerdProfiles();
  const { activePreset, dataset: uploadedDataset } = useUploadedCows();
  const {
    statsArray: activePresetStats,
    isLoading: activePresetStatsLoading,
    isError: activePresetStatsError,
  } = usePresetHerdStats(
    activePreset?.dataset ?? null,
    activePreset?.size ?? "small",
    activePreset?.period ?? "mixed"
  );
  const createMutation = useCreateHerdProfile();
  const updateMutation = useUpdateHerdProfile();
  const deleteMutation = useDeleteHerdProfile();

  const [createOpen, setCreateOpen] = useState(false);
  const [createMode, setCreateMode] = useState<"manual" | "dataset">("manual");
  const [editTarget, setEditTarget] = useState<HerdProfile | null>(null);
  const [createdProfileName, setCreatedProfileName] = useState<string | null>(null);

  const activeDatasetLabel = activePreset
    ? `${PRESET_LABELS[activePreset.dataset]} · ${PERIOD_LABELS[activePreset.period]} · ${SIZE_LABELS[activePreset.size]} sample`
    : uploadedDataset
      ? `${uploadedDataset.name} · ${uploadedDataset.cows.length.toLocaleString()} lactations`
      : null;

  const datasetFormStats =
    createMode === "dataset" && activePresetStats ? activePresetStats : [...DEFAULT_HERD_STATS];

  function handleCreate(data: HerdProfileCreate) {
    createMutation.mutate(data, {
      onSuccess: (created) => {
        setCreateOpen(false);
        setCreatedProfileName(created.name);
      },
    });
  }

  function handleUpdate(data: HerdProfileCreate) {
    if (!editTarget) return;
    updateMutation.mutate({ id: editTarget.id, data }, { onSuccess: () => setEditTarget(null) });
  }

  function handleDelete(profile: HerdProfile) {
    if (confirm(`Delete profile "${profile.name}"?`)) {
      deleteMutation.mutate(profile.id);
    }
  }

  if (isLoading) return <Text>Loading profiles…</Text>;

  return (
    <>
      <Stack gap="md">
        <Alert color={activeDatasetLabel ? "violet" : "gray"} variant="light">
          <Group justify="space-between" align="center" gap="sm">
            <div>
              <Text size="xs" fw={700} tt="uppercase" c="dimmed">
                Active dataset
              </Text>
              <Text size="sm">
                {activeDatasetLabel ??
                  "No dataset selected. Load a preset or upload a file in Data Upload first."}
              </Text>
            </div>
            <Button
              component={Link}
              href="/data-upload"
              variant="subtle"
              size="xs"
              rightSection={<ChevronRight size={14} />}
            >
              Data Upload
            </Button>
          </Group>
        </Alert>

        <Group justify="space-between" align="flex-start">
          <div>
            <Text fw={500}>Saved Herd Profiles</Text>
            <Text size="xs" c="dimmed">
              Select the desired herd statistics while fitting the lactation curve with the
              autoencoder.
            </Text>
          </div>
          <Group gap="xs">
            <Button
              size="sm"
              variant="light"
              disabled={!activePreset || !activePresetStats || activePresetStatsError}
              loading={activePresetStatsLoading}
              onClick={() => {
                setCreateMode("dataset");
                setCreateOpen(true);
              }}
            >
              Determine from dataset
            </Button>
            <Button
              size="sm"
              color="violet"
              onClick={() => {
                setCreateMode("manual");
                setCreateOpen(true);
              }}
            >
              New profile
            </Button>
          </Group>
        </Group>

        {uploadedDataset && !activePreset && (
          <Text size="xs" c="dimmed">
            Uploaded files can be saved as profiles from the upload preview on the Data Upload page.
          </Text>
        )}

        {profiles.length === 0 ? (
          <Text size="sm">No profiles yet. Create one to save a set of herd statistics.</Text>
        ) : (
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Description</Table.Th>
                <Table.Th>Created</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {profiles.map((profile) => (
                <Table.Tr key={profile.id}>
                  <Table.Td>{profile.name}</Table.Td>
                  <Table.Td>{profile.description || "-"}</Table.Td>
                  <Table.Td>
                    {profile.created_at ? new Date(profile.created_at).toLocaleDateString() : "-"}
                  </Table.Td>
                  <Table.Td>
                    <Group gap="xs" justify="flex-end">
                      <Button component={Link} href="/curves" variant="subtle" size="xs">
                        Use in Curves
                      </Button>
                      <ActionIcon
                        variant="subtle"
                        onClick={() => setEditTarget(profile)}
                        aria-label="Edit profile"
                      >
                        <Pencil size={14} />
                      </ActionIcon>
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        onClick={() => handleDelete(profile)}
                        aria-label="Delete profile"
                        loading={deleteMutation.isPending}
                      >
                        <Trash2 size={14} />
                      </ActionIcon>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        )}
      </Stack>

      <Modal
        opened={createOpen}
        onClose={() => setCreateOpen(false)}
        title={createMode === "dataset" ? "New profile from active dataset" : "New herd profile"}
        size="xl"
      >
        <HerdProfileForm
          key={`${createMode}-${activeDatasetLabel ?? "manual"}`}
          initialStats={datasetFormStats}
          defaultName={
            createMode === "dataset" && activePreset
              ? `${PRESET_LABELS[activePreset.dataset]} - ${PERIOD_LABELS[activePreset.period]} (${SIZE_LABELS[activePreset.size]})`
              : undefined
          }
          defaultDescription={
            createMode === "dataset" && activePreset
              ? `Herd statistics derived from ${PRESET_LABELS[activePreset.dataset]} (${PERIOD_LABELS[activePreset.period].toLowerCase()} period, ${SIZE_LABELS[activePreset.size].toLowerCase()} sample).`
              : undefined
          }
          sourceSummary={
            createMode === "dataset" && activeDatasetLabel
              ? `Values are determined from ${activeDatasetLabel}.`
              : undefined
          }
          onSubmit={handleCreate}
          onCancel={() => setCreateOpen(false)}
          isLoading={createMutation.isPending}
        />
      </Modal>

      <Modal
        opened={createdProfileName !== null}
        onClose={() => setCreatedProfileName(null)}
        title="Herd profile created"
        size="md"
      >
        <Stack gap="md">
          <Alert icon={<CheckCircle2 size={16} />} color="green">
            <Text size="sm">
              {createdProfileName} is saved. You can use this profile when fitting lactation curves
              with the autoencoder.
            </Text>
          </Alert>
          <Group justify="flex-end">
            <Button variant="subtle" onClick={() => setCreatedProfileName(null)}>
              Close
            </Button>
            <Button
              component={Link}
              href="/curves"
              color="violet"
              rightSection={<ChevronRight size={14} />}
            >
              Go to Curves
            </Button>
          </Group>
        </Stack>
      </Modal>

      <Modal
        opened={editTarget !== null}
        onClose={() => setEditTarget(null)}
        title="Edit herd profile"
        size="xl"
      >
        {editTarget && (
          <HerdProfileForm
            initial={editTarget}
            onSubmit={handleUpdate}
            onCancel={() => setEditTarget(null)}
            isLoading={updateMutation.isPending}
          />
        )}
      </Modal>
    </>
  );
}
