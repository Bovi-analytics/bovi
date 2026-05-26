"use client";

import type { ReactElement } from "react";
import { Group, SegmentedControl, Select, Stack, Tabs, Text } from "@mantine/core";
import type { BenchmarkModel, MilkBotRunOptions } from "@/types/api";

type BenchmarkModelGroup = "yield" | "curve" | "deep";

interface BenchmarkModelOption {
  readonly value: BenchmarkModel;
  readonly label: string;
}

const MODEL_GROUPS: Record<
  BenchmarkModelGroup,
  {
    readonly label: string;
    readonly description: string;
    readonly models: readonly BenchmarkModelOption[];
  }
> = {
  yield: {
    label: "305-day yield",
    description: "Direct ALY estimators from test-day records.",
    models: [
      { value: "tim", label: "TIM (Test Interval Method)" },
      { value: "islc", label: "ISLC" },
      { value: "best_predict", label: "Best Predict" },
    ],
  },
  curve: {
    label: "Curve fit",
    description: "Classical lactation curves scored by cumulative yield.",
    models: [
      { value: "wood", label: "Wood" },
      { value: "wilmink", label: "Wilmink" },
      { value: "ali_schaeffer", label: "Ali-Schaeffer" },
      { value: "fischer", label: "Fischer" },
      { value: "milkbot", label: "MilkBot" },
    ],
  },
  deep: {
    label: "Deep learning",
    description: "Full-curve neural prediction scored by cumulative yield.",
    models: [{ value: "autoencoder", label: "AI autoencoder" }],
  },
};

const MODEL_TO_GROUP = Object.entries(MODEL_GROUPS).reduce(
  (acc, [group, config]) => {
    config.models.forEach((model) => {
      acc[model.value] = group as BenchmarkModelGroup;
    });
    return acc;
  },
  {} as Record<BenchmarkModel, BenchmarkModelGroup>
);

interface Props {
  readonly label: string;
  readonly value: BenchmarkModel;
  readonly onChange: (value: BenchmarkModel) => void;
  readonly milkbotOptions: MilkBotRunOptions;
  readonly onMilkbotOptionsChange: (options: MilkBotRunOptions) => void;
}

export function BenchmarkModelPicker({
  label,
  value,
  onChange,
  milkbotOptions,
  onMilkbotOptionsChange,
}: Props): ReactElement {
  const activeGroup = MODEL_TO_GROUP[value];
  const activeModels = MODEL_GROUPS[activeGroup].models;

  function updateMilkBotOptions(patch: Partial<MilkBotRunOptions>): void {
    onMilkbotOptionsChange({ ...milkbotOptions, ...patch });
  }

  function handleTabChange(nextGroup: string | null): void {
    if (!nextGroup || nextGroup === activeGroup) {
      return;
    }
    const group = nextGroup as BenchmarkModelGroup;
    onChange(MODEL_GROUPS[group].models[0].value);
  }

  return (
    <Stack gap={6}>
      <Tabs value={activeGroup} onChange={handleTabChange}>
        <Tabs.List grow>
          {Object.entries(MODEL_GROUPS).map(([group, config]) => (
            <Tabs.Tab key={group} value={group}>
              {config.label}
            </Tabs.Tab>
          ))}
        </Tabs.List>
      </Tabs>
      <Select
        label={label}
        data={activeModels}
        value={value}
        onChange={(nextValue) => nextValue && onChange(nextValue as BenchmarkModel)}
      />
      <Text size="xs" c="dimmed">
        {MODEL_GROUPS[activeGroup].description}
      </Text>
      {value === "milkbot" && (
        <Stack gap={6}>
          <SegmentedControl
            size="xs"
            value={milkbotOptions.fitting}
            onChange={(nextValue) =>
              updateMilkBotOptions({ fitting: nextValue as MilkBotRunOptions["fitting"] })
            }
            data={[
              { value: "frequentist", label: "Frequentist" },
              { value: "bayesian", label: "Bayesian" },
            ]}
          />
          <Group grow>
            <Select
              size="xs"
              label="Prior source"
              value={milkbotOptions.continent}
              onChange={(nextValue) =>
                nextValue &&
                updateMilkBotOptions({ continent: nextValue as MilkBotRunOptions["continent"] })
              }
              data={[
                { value: "USA", label: "USA" },
                { value: "EU", label: "EU" },
                { value: "CHEN", label: "Chen et al." },
              ]}
              allowDeselect={false}
            />
            <Select
              size="xs"
              label="Breed"
              value={milkbotOptions.breed}
              onChange={(nextValue) =>
                nextValue &&
                updateMilkBotOptions({ breed: nextValue as MilkBotRunOptions["breed"] })
              }
              data={[
                { value: "H", label: "Holstein" },
                { value: "J", label: "Jersey" },
              ]}
              allowDeselect={false}
            />
          </Group>
          <Text size="xs" c="dimmed">
            Parity is read per lactation from the reference dataset.
          </Text>
        </Stack>
      )}
    </Stack>
  );
}
