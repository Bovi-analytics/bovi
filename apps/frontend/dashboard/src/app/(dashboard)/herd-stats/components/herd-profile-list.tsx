"use client";

import type { ReactElement } from "react";
import { useState } from "react";
import {
  ActionIcon,
  Alert,
  Button,
  Group,
  Modal,
  Select,
  SegmentedControl,
  Stack,
  Table,
  Text,
  TextInput,
} from "@mantine/core";
import { Pencil, Trash2 } from "lucide-react";
import type { OrganizationListOptions } from "@/lib/api-client";
import { useAuth } from "@/lib/auth";
import { HerdProfileForm } from "./herd-profile-form";
import {
  useCreateHerdProfile,
  useDeleteHerdProfile,
  useHerdProfiles,
  useUpdateHerdProfile,
} from "../hooks/use-herd-profiles";
import type { HerdProfile, HerdProfileCreate } from "@/types/api";
import { useUploadedCows } from "@/app/providers/uploaded-cows-provider";

const PRESET_LABELS: Record<string, string> = {
  aurora: "Preset cohort A",
  sunnyside: "Preset cohort B",
};
const PERIOD_LABELS: Record<string, string> = { recent: "Recent", old: "Old", mixed: "Mixed" };
const SIZE_LABELS: Record<string, string> = { small: "Small", medium: "Medium", large: "Large" };

export function HerdProfileList(): ReactElement {
  const { selectedOrganizationId } = useAuth();
  const [scope, setScope] = useState<"organization" | "mine">("organization");
  const [sort, setSort] = useState<"created_at" | "name" | "user">("created_at");
  const [direction, setDirection] = useState<"asc" | "desc">("desc");
  const [q, setQ] = useState("");
  const options: OrganizationListOptions = { scope, sort, direction, q: q.trim() || undefined };
  const { data: profiles = [], isLoading } = useHerdProfiles(options);
  const { activePreset } = useUploadedCows();
  const createMutation = useCreateHerdProfile();
  const updateMutation = useUpdateHerdProfile();
  const deleteMutation = useDeleteHerdProfile();

  const [createOpen, setCreateOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<HerdProfile | null>(null);
  const createDisabled = selectedOrganizationId === "all";

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

  if (isLoading) return <Text>Loading profiles…</Text>;

  return (
    <>
      <Stack gap="md">
        <Group justify="space-between">
          <Text fw={500}>Saved Herd Profiles</Text>
          <Button
            size="sm"
            color="violet"
            disabled={createDisabled}
            onClick={() => setCreateOpen(true)}
          >
            New profile
          </Button>
        </Group>

        {createDisabled && (
          <Alert color="yellow" variant="light">
            Select a specific organization before creating a herd profile.
          </Alert>
        )}

        <Group gap="sm" align="flex-end">
          <SegmentedControl
            size="xs"
            value={scope}
            onChange={(value) => setScope(value as "organization" | "mine")}
            data={[
              { label: "Organization", value: "organization" },
              { label: "My items", value: "mine" },
            ]}
          />
          <TextInput
            aria-label="Search herd profiles"
            placeholder="Search by name"
            value={q}
            onChange={(event) => setQ(event.currentTarget.value)}
            size="xs"
          />
          <Select
            aria-label="Sort herd profiles"
            size="xs"
            value={sort}
            onChange={(value) =>
              setSort((value as "created_at" | "name" | "user") ?? "created_at")
            }
            data={[
              { label: "Created", value: "created_at" },
              { label: "Name", value: "name" },
              { label: "User", value: "user" },
            ]}
          />
          <Select
            aria-label="Sort direction"
            size="xs"
            value={direction}
            onChange={(value) => setDirection((value as "asc" | "desc") ?? "desc")}
            data={[
              { label: "Newest first", value: "desc" },
              { label: "Oldest first", value: "asc" },
            ]}
          />
        </Group>

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
              ? `${PRESET_LABELS[activePreset.dataset]} - ${PERIOD_LABELS[activePreset.period]} (${SIZE_LABELS[activePreset.size]})`
              : undefined
          }
          defaultDescription={
            activePreset
              ? `Herd statistics derived from ${PRESET_LABELS[activePreset.dataset]} (${PERIOD_LABELS[activePreset.period].toLowerCase()} period, ${SIZE_LABELS[activePreset.size].toLowerCase()} sample).`
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
