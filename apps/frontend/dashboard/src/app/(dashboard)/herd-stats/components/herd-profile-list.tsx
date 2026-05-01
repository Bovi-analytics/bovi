"use client";

import type { ReactElement } from "react";
import { useState } from "react";
import { ActionIcon, Button, Group, Modal, Stack, Table, Text } from "@mantine/core";
import { Pencil, Trash2 } from "lucide-react";
import { HerdProfileForm } from "./herd-profile-form";
import {
  useCreateHerdProfile,
  useDeleteHerdProfile,
  useHerdProfiles,
  useUpdateHerdProfile,
} from "../hooks/use-herd-profiles";
import type { HerdProfile, HerdProfileCreate } from "@/types/api";
import { useUploadedCows } from "@/app/providers/uploaded-cows-provider";

const PRESET_LABELS: Record<string, string> = { aurora: "Aurora Ridge", sunnyside: "Sunnyside" };
const PERIOD_LABELS: Record<string, string> = { recent: "Recent", old: "Old", mixed: "Mixed" };
const SIZE_LABELS: Record<string, string> = { small: "Small", medium: "Medium", large: "Large" };

export function HerdProfileList(): ReactElement {
  const { data: profiles = [], isLoading } = useHerdProfiles();
  const { activePreset } = useUploadedCows();
  const createMutation = useCreateHerdProfile();
  const updateMutation = useUpdateHerdProfile();
  const deleteMutation = useDeleteHerdProfile();

  const [createOpen, setCreateOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<HerdProfile | null>(null);

  function handleCreate(data: HerdProfileCreate) {
    createMutation.mutate(data, { onSuccess: () => setCreateOpen(false) });
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

  if (isLoading) return <Text c="dimmed">Loading profiles…</Text>;

  return (
    <>
      <Stack gap="md">
        <Group justify="space-between">
          <Text fw={500}>Saved Herd Profiles</Text>
          <Button size="sm" color="violet" onClick={() => setCreateOpen(true)}>
            New profile
          </Button>
        </Group>

        {profiles.length === 0 ? (
          <Text c="dimmed" size="sm">
            No profiles yet. Create one to save a set of herd statistics.
          </Text>
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
                  <Table.Td c="dimmed">{profile.description || "—"}</Table.Td>
                  <Table.Td>
                    {profile.created_at
                      ? new Date(profile.created_at).toLocaleDateString()
                      : "—"}
                  </Table.Td>
                  <Table.Td>
                    <Group gap="xs" justify="flex-end">
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
        title="New herd profile"
        size="xl"
      >
        <HerdProfileForm
          defaultName={
            activePreset
              ? `${PRESET_LABELS[activePreset.dataset]} — ${PERIOD_LABELS[activePreset.period]} (${SIZE_LABELS[activePreset.size]})`
              : undefined
          }
          defaultDescription={
            activePreset
              ? `Herd statistics derived from the ${PRESET_LABELS[activePreset.dataset]} preset dataset (${PERIOD_LABELS[activePreset.period].toLowerCase()} period, ${SIZE_LABELS[activePreset.size].toLowerCase()} sample).`
              : undefined
          }
          onSubmit={handleCreate}
          onCancel={() => setCreateOpen(false)}
          isLoading={createMutation.isPending}
        />
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
