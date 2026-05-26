"use client";

import type { ReactElement } from "react";
import {
  Alert,
  Badge,
  Group,
  Loader,
  Paper,
  SegmentedControl,
  Stack,
  Text,
  UnstyledButton,
} from "@mantine/core";
import { AlertCircle } from "lucide-react";
import { useUploadedCows } from "@/app/providers/uploaded-cows-provider";
import { usePresetCounts } from "@/app/(dashboard)/curves/hooks/use-preset-counts";
import { usePresetDataset } from "@/app/(dashboard)/curves/hooks/use-preset-dataset";
import type { PresetDatasetKey, PresetPeriodKey, PresetSizeKey } from "@/types/api";

const DATASET_OPTIONS: { value: PresetDatasetKey | "none"; label: string; description: string }[] =
  [
    { value: "none", label: "None", description: "Use uploaded CSV or manual input" },
    { value: "aurora", label: "Demo herd A", description: "Anonymized herd · 2023-2025" },
    { value: "sunnyside", label: "Demo herd B", description: "Anonymized herd · 2000-2026" },
  ];

const SIZE_OPTIONS = [
  { value: "small", label: "Small" },
  { value: "medium", label: "Medium" },
  { value: "large", label: "Large" },
] satisfies Array<{ value: PresetSizeKey; label: string }>;

const PERIOD_OPTIONS = [
  { value: "recent", label: "Recent" },
  { value: "old", label: "Old" },
  { value: "mixed", label: "Mixed" },
];

function sizeOptionsWithCounts(
  counts: Record<string, Record<string, number>> | undefined,
  period: PresetPeriodKey
): Array<{ value: PresetSizeKey; label: string }> {
  return SIZE_OPTIONS.map((option) => {
    const count = counts?.[period]?.[option.value];
    return {
      ...option,
      label: count === undefined ? option.label : `${option.label} (${count.toLocaleString()})`,
    };
  });
}

export function PresetHerdPicker(): ReactElement {
  const { activePreset, setActivePreset } = useUploadedCows();

  const selectedDataset = activePreset?.dataset ?? null;
  const selectedSize = activePreset?.size ?? "small";
  const selectedPeriod = activePreset?.period ?? "mixed";

  const {
    data: presetData,
    isLoading,
    isError,
  } = usePresetDataset(selectedDataset, selectedSize, selectedPeriod);
  const { data: presetCounts } = usePresetCounts(selectedDataset);

  function handleSelectDataset(value: PresetDatasetKey | "none") {
    if (value === "none") {
      setActivePreset(null);
    } else {
      setActivePreset({ dataset: value, size: selectedSize, period: selectedPeriod });
    }
  }

  function handleSizeChange(size: string) {
    if (!selectedDataset) return;
    setActivePreset({
      dataset: selectedDataset,
      size: size as PresetSizeKey,
      period: selectedPeriod,
    });
  }

  function handlePeriodChange(period: string) {
    if (!selectedDataset) return;
    setActivePreset({
      dataset: selectedDataset,
      size: selectedSize,
      period: period as PresetPeriodKey,
    });
  }

  return (
    <Stack gap="sm">
      <Text size="sm" fw={500}>
        Demo herds
      </Text>
      <Text size="xs">
        Pick a built-in demo herd as the active herd. The selected lactations will be available in
        the Curves tab for individual lactation analysis.
      </Text>

      <Group gap="sm">
        {DATASET_OPTIONS.map((opt) => {
          const isActive = (selectedDataset ?? "none") === opt.value;
          return (
            <UnstyledButton key={opt.value} onClick={() => handleSelectDataset(opt.value)}>
              <Paper
                withBorder
                p="sm"
                radius="md"
                style={{
                  borderColor: isActive ? "var(--mantine-color-violet-6)" : undefined,
                  borderWidth: isActive ? 2 : 1,
                  minWidth: 160,
                  cursor: "pointer",
                  transition: "border-color 0.1s",
                }}
              >
                <Text size="sm" fw={600}>
                  {opt.label}
                </Text>
                <Text size="xs">{opt.description}</Text>
              </Paper>
            </UnstyledButton>
          );
        })}
      </Group>

      {selectedDataset && (
        <Stack gap="xs">
          <Group gap="xl" align="flex-start">
            <Stack gap={4}>
              <Text size="xs">Sample size</Text>
              <SegmentedControl
                size="xs"
                value={selectedSize}
                onChange={handleSizeChange}
                data={sizeOptionsWithCounts(presetCounts?.counts, selectedPeriod)}
              />
            </Stack>
            <Stack gap={4}>
              <Text size="xs">Time period</Text>
              <SegmentedControl
                size="xs"
                value={selectedPeriod}
                onChange={handlePeriodChange}
                data={PERIOD_OPTIONS}
              />
            </Stack>
            <Stack gap={4} justify="flex-end" style={{ paddingTop: 20 }}>
              {isLoading && <Loader size="xs" />}
              {presetData && !isLoading && (
                <Badge color="violet" variant="light">
                  {presetData.cow_count.toLocaleString()} lactations loaded
                </Badge>
              )}
            </Stack>
          </Group>
          {isError && (
            <Alert icon={<AlertCircle size={14} />} color="red" p="xs">
              Dataset unavailable - make sure CONNECTION_STRING is configured and the preprocessing
              script has been run.
            </Alert>
          )}
        </Stack>
      )}
    </Stack>
  );
}
