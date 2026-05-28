"use client";

import type { ReactElement } from "react";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
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
import { CheckCircle2, ChevronRight, Pencil, Trash2 } from "lucide-react";
import { listOrganizationMembers, type OrganizationListOptions } from "@/lib/api-client";
import { useAuth } from "@/lib/auth";
import { HerdProfileForm } from "./herd-profile-form";
import {
  ActiveDatasetPanel,
  useActiveDatasetLabel,
} from "@/components/dashboard/active-dataset-panel";
import {
  useCreateHerdProfile,
  useDeleteHerdProfile,
  useHerdProfiles,
  useUpdateHerdProfile,
} from "../hooks/use-herd-profiles";
import type { HerdProfile, HerdProfileCreate } from "@/types/api";
import { useUploadedCows } from "@/app/providers/uploaded-cows-provider";
import { DEFAULT_HERD_STATS, HERD_STATS_METADATA } from "@/data/herd-stats-metadata";
import { usePresetHerdStats } from "@/app/(dashboard)/curves/hooks/use-preset-herd-stats";

const PRESET_LABELS: Record<string, string> = {
  aurora: "Demo herd A",
  sunnyside: "Demo herd B",
};
const PERIOD_LABELS: Record<string, string> = { recent: "Recent", old: "Old", mixed: "Mixed" };
const SIZE_LABELS: Record<string, string> = { small: "Small", medium: "Medium", large: "Large" };

function profileSource(profile: HerdProfile): string {
  const description = profile.description.toLowerCase();
  if (description.includes("uploaded dataset")) return "Uploaded dataset";
  if (description.includes("demo herd")) return "Demo dataset";
  if (description.includes("derived from")) return "Dataset";
  return "Manual";
}

export function HerdProfileList(): ReactElement {
  const { selectedOrganizationId } = useAuth();
  const [scope, setScope] = useState<"organization" | "mine" | "colleague">("organization");
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);
  const [sort, setSort] = useState<"created_at" | "name" | "user">("created_at");
  const [direction, setDirection] = useState<"asc" | "desc">("desc");
  const [q, setQ] = useState("");
  const options: OrganizationListOptions = {
    scope: scope === "mine" ? "mine" : "organization",
    user_id: scope === "colleague" && selectedUserId ? Number(selectedUserId) : undefined,
    sort,
    direction,
    q: q.trim() || undefined,
  };
  const { data: profiles = [], isLoading } = useHerdProfiles(options);
  const membersQuery = useQuery({
    queryKey: ["organization-members", selectedOrganizationId],
    queryFn: () => listOrganizationMembers(selectedOrganizationId as number),
    enabled: typeof selectedOrganizationId === "number",
  });
  const { activePreset, dataset: uploadedDataset } = useUploadedCows();
  const activeDatasetLabel = useActiveDatasetLabel();
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
  const createDisabled = selectedOrganizationId === "all";
  const [createdProfileName, setCreatedProfileName] = useState<string | null>(null);

  const uploadedDatasetStats = uploadedDataset?.stats
    ? HERD_STATS_METADATA.map((meta) => uploadedDataset.stats?.[meta.name] ?? meta.default)
    : null;

  const datasetFormStats =
    createMode === "dataset" && activePresetStats
      ? activePresetStats
      : createMode === "dataset" && uploadedDatasetStats
        ? uploadedDatasetStats
        : [...DEFAULT_HERD_STATS];

  const canDetermineFromDataset = Boolean(
    (activePreset && activePresetStats && !activePresetStatsError) ||
    (!activePreset && uploadedDatasetStats)
  );

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
        <ActiveDatasetPanel
          emptyText="No dataset selected. Load a preset or upload a file in Data Upload first."
          actionHref="/data-upload"
          actionLabel={activeDatasetLabel ? "Change" : "Data Upload"}
          showActionWithoutDataset
        />

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
              disabled={createDisabled || !canDetermineFromDataset}
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
              disabled={createDisabled}
              onClick={() => {
                setCreateMode("manual");
                setCreateOpen(true);
              }}
            >
              New profile
            </Button>
          </Group>
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
            onChange={(value) => setScope(value as "organization" | "mine" | "colleague")}
            data={[
              { label: "Organization", value: "organization" },
              { label: "My items", value: "mine" },
              { label: "Colleague", value: "colleague" },
            ]}
          />
          {scope === "colleague" && (
            <Select
              aria-label="Filter by colleague"
              size="xs"
              value={selectedUserId}
              onChange={setSelectedUserId}
              placeholder="Select colleague"
              data={(membersQuery.data ?? []).map((member) => ({
                value: String(member.user_id),
                label: member.name || member.email || `User #${member.user_id}`,
              }))}
            />
          )}
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
            onChange={(value) => setSort((value as "created_at" | "name" | "user") ?? "created_at")}
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
                <Table.Th>Source</Table.Th>
                <Table.Th>Owner</Table.Th>
                <Table.Th>Description</Table.Th>
                <Table.Th>Created</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {profiles.map((profile) => (
                <Table.Tr key={profile.id}>
                  <Table.Td>{profile.name}</Table.Td>
                  <Table.Td>{profileSource(profile)}</Table.Td>
                  <Table.Td>
                    {profile.user_name || profile.user_email || `User #${profile.user_id}`}
                  </Table.Td>
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
              : createMode === "dataset" && uploadedDataset
                ? `${uploadedDataset.name} profile`
                : undefined
          }
          defaultDescription={
            createMode === "dataset" && activePreset
              ? `Herd statistics derived from ${PRESET_LABELS[activePreset.dataset]} (${PERIOD_LABELS[activePreset.period].toLowerCase()} period, ${SIZE_LABELS[activePreset.size].toLowerCase()} sample).`
              : createMode === "dataset" && uploadedDataset
                ? `Herd statistics derived from uploaded dataset ${uploadedDataset.name}.`
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
