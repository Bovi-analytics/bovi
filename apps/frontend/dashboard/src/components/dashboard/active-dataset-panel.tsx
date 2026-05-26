"use client";

import type { ReactElement } from "react";
import Link from "next/link";
import { Alert, Badge, Button, Group, Text } from "@mantine/core";
import { ChevronRight } from "lucide-react";
import { useUploadedCows } from "@/app/providers/uploaded-cows-provider";
import type { PresetDatasetKey } from "@/types/api";

const PRESET_LABELS: Record<PresetDatasetKey, string> = {
  aurora: "Demo herd A",
  sunnyside: "Demo herd B",
};

const PERIOD_LABELS = {
  recent: "Recent",
  old: "Old",
  mixed: "Mixed",
};

const SIZE_LABELS = {
  small: "Small",
  medium: "Medium",
  large: "Large",
};

interface ActiveDatasetPanelProps {
  readonly emptyText?: string;
  readonly actionHref?: string;
  readonly actionLabel?: string;
  readonly showActionWithoutDataset?: boolean;
  readonly compact?: boolean;
}

export function useActiveDatasetLabel(): string | null {
  const { activePreset, dataset } = useUploadedCows();
  if (activePreset) {
    return `${PRESET_LABELS[activePreset.dataset]} · ${PERIOD_LABELS[activePreset.period]} · ${SIZE_LABELS[activePreset.size]} sample`;
  }
  if (dataset) {
    return `${dataset.name} · ${(dataset.cowCount ?? dataset.cows.length).toLocaleString()} lactations`;
  }
  return null;
}

export function ActiveDatasetPanel({
  emptyText = "No dataset selected yet.",
  actionHref,
  actionLabel,
  showActionWithoutDataset = false,
  compact = false,
}: ActiveDatasetPanelProps): ReactElement {
  const label = useActiveDatasetLabel();
  const showAction = actionHref && actionLabel && (label || showActionWithoutDataset);

  return (
    <Alert color={label ? "violet" : "gray"} variant="light" py={compact ? "xs" : undefined}>
      <Group justify="space-between" align="center" gap="sm">
        <div>
          <Text size="xs" fw={700} tt="uppercase" c="dimmed">
            Active dataset
          </Text>
          <Text size="sm">{label ?? emptyText}</Text>
        </div>
        <Group gap="xs">
          {label && (
            <Badge color="violet" variant="filled">
              Selected
            </Badge>
          )}
          {showAction && (
            <Button
              component={Link}
              href={actionHref}
              variant={label ? "light" : "subtle"}
              size="xs"
              rightSection={<ChevronRight size={14} />}
            >
              {actionLabel}
            </Button>
          )}
        </Group>
      </Group>
    </Alert>
  );
}
