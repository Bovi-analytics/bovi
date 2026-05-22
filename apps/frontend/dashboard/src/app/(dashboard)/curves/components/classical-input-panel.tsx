"use client";

import { useState } from "react";
import type { ReactElement } from "react";
import { Checkbox, Group, Modal, SegmentedControl, Select, Stack, Text } from "@mantine/core";
import { Info } from "lucide-react";
import { ALL_MODELS, MODEL_METADATA } from "@/data/model-metadata";
import { ModelInfo } from "@/app/(dashboard)/models/components/model-info";
import type { MilkBotRunOptions, Model } from "@/types/api";
import type { ModelMetadata } from "@/data/model-metadata";

interface ClassicalInputPanelProps {
  readonly selectedModels: Model[];
  readonly onToggleModel: (model: Model) => void;
  readonly parity: number;
  readonly milkbotOptions: MilkBotRunOptions;
  readonly onMilkbotOptionsChange: (options: MilkBotRunOptions) => void;
}

export function ClassicalInputPanel({
  selectedModels,
  onToggleModel,
  parity,
  milkbotOptions,
  onMilkbotOptionsChange,
}: ClassicalInputPanelProps): ReactElement {
  const [modalModel, setModalModel] = useState<ModelMetadata | null>(null);
  const showMilkBotOptions = selectedModels.includes("milkbot");

  function updateMilkBotOptions(patch: Partial<MilkBotRunOptions>): void {
    onMilkbotOptionsChange({ ...milkbotOptions, ...patch });
  }

  return (
    <>
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-base font-semibold text-foreground">Classical models</h3>
        <div className="space-y-2">
          {ALL_MODELS.map((m) => (
            <div key={m.id} className="flex items-center justify-between">
              <Checkbox
                label={m.name}
                checked={selectedModels.includes(m.id)}
                onChange={() => onToggleModel(m.id)}
                color={m.color}
                size="sm"
              />
              <button
                type="button"
                onClick={() => setModalModel(MODEL_METADATA[m.id])}
                className="ml-2 text-muted-foreground/50 transition-colors hover:text-muted-foreground"
                aria-label={`Info about ${m.name}`}
              >
                <Info size={14} />
              </button>
            </div>
          ))}
        </div>
        {showMilkBotOptions && (
          <Stack gap="xs" mt="md">
            <Text size="xs" fw={700} tt="uppercase" c="dimmed">
              MilkBot fitting
            </Text>
            <SegmentedControl
              size="xs"
              value={milkbotOptions.fitting}
              onChange={(value) =>
                updateMilkBotOptions({ fitting: value as MilkBotRunOptions["fitting"] })
              }
              data={[
                { value: "frequentist", label: "Frequentist" },
                { value: "bayesian", label: "Bayesian" },
              ]}
            />
            <Group grow align="start">
              <Select
                size="xs"
                label="Prior source"
                value={milkbotOptions.continent}
                onChange={(value) =>
                  value &&
                  updateMilkBotOptions({ continent: value as MilkBotRunOptions["continent"] })
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
                onChange={(value) =>
                  value && updateMilkBotOptions({ breed: value as MilkBotRunOptions["breed"] })
                }
                data={[
                  { value: "H", label: "Holstein" },
                  { value: "J", label: "Jersey" },
                ]}
                allowDeselect={false}
              />
            </Group>
            <Text size="xs" c="dimmed">
              Parity {parity} is sent with Bayesian MilkBot requests.
            </Text>
          </Stack>
        )}
      </div>

      <Modal
        opened={modalModel !== null}
        onClose={() => setModalModel(null)}
        title={modalModel?.name ?? ""}
        size="lg"
      >
        {modalModel && <ModelInfo model={modalModel} />}
      </Modal>
    </>
  );
}
