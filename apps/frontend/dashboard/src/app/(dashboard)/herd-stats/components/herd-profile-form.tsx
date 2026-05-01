"use client";

import type { ReactElement } from "react";
import { useState } from "react";
import { Button, Group, Stack, Textarea, TextInput } from "@mantine/core";
import { HerdStatsForm } from "@/app/(dashboard)/autoencoder/components/herd-stats-form";
import { DEFAULT_HERD_STATS } from "@/data/herd-stats-metadata";
import { herdProfileToStats, statsToHerdProfileFields } from "@/lib/herd-profile-utils";
import type { HerdProfile, HerdProfileCreate } from "@/types/api";

interface HerdProfileFormProps {
  readonly initial?: HerdProfile;
  readonly defaultName?: string;
  readonly defaultDescription?: string;
  readonly onSubmit: (data: HerdProfileCreate) => void;
  readonly onCancel: () => void;
  readonly isLoading: boolean;
}

export function HerdProfileForm({
  initial,
  defaultName,
  defaultDescription,
  onSubmit,
  onCancel,
  isLoading,
}: HerdProfileFormProps): ReactElement {
  const [name, setName] = useState(initial?.name ?? defaultName ?? "");
  const [description, setDescription] = useState(initial?.description ?? defaultDescription ?? "");
  const [stats, setStats] = useState<number[]>(
    initial ? herdProfileToStats(initial) : [...DEFAULT_HERD_STATS]
  );

  function handleSubmit() {
    onSubmit({
      name,
      description,
      ...statsToHerdProfileFields(stats),
    });
  }

  return (
    <Stack gap="md">
      <TextInput
        label="Profile name"
        placeholder="e.g. High-producing Holstein"
        value={name}
        onChange={(e) => setName(e.currentTarget.value)}
        maxLength={100}
        required
      />
      <Textarea
        label="Description"
        placeholder="Optional notes about this herd"
        value={description}
        onChange={(e) => setDescription(e.currentTarget.value)}
        maxLength={500}
        autosize
        minRows={2}
      />
      <div>
        <p className="mb-3 text-xs text-muted-foreground">All values normalized 0–1.</p>
        <HerdStatsForm values={stats} onChange={setStats} />
      </div>
      <Group justify="flex-end">
        <Button variant="subtle" onClick={onCancel} disabled={isLoading}>
          Cancel
        </Button>
        <Button
          onClick={handleSubmit}
          loading={isLoading}
          disabled={!name.trim()}
          color="violet"
        >
          {initial ? "Save changes" : "Create profile"}
        </Button>
      </Group>
    </Stack>
  );
}
