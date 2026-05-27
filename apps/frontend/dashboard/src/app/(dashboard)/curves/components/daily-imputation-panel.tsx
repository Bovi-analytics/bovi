"use client";

import type { ReactElement } from "react";
import { Select, Stack, Switch, Text, Tooltip } from "@mantine/core";
import { Info } from "lucide-react";
import type { ImputationMethod } from "@/types/api";

const IMPUTATION_OPTIONS = [
  {
    value: "forward_fill",
    label: "Forward fill",
    description: "Uses the most recent known milk value for later missing days.",
  },
  {
    value: "backward_fill",
    label: "Backward fill",
    description: "Uses the next known milk value for earlier missing days.",
  },
  {
    value: "linear",
    label: "Linear interpolation",
    description: "Draws a straight line between neighboring known values.",
  },
] as const;

interface DailyImputationPanelProps {
  readonly useImputation: boolean;
  readonly onUseImputationChange: (useImputation: boolean) => void;
  readonly imputationMethod: ImputationMethod;
  readonly onImputationMethodChange: (method: ImputationMethod) => void;
}

export function DailyImputationPanel({
  useImputation,
  onUseImputationChange,
  imputationMethod,
  onImputationMethodChange,
}: DailyImputationPanelProps): ReactElement {
  const selectedImputationOption =
    IMPUTATION_OPTIONS.find((option) => option.value === imputationMethod) ?? IMPUTATION_OPTIONS[0];

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-3 text-base font-semibold text-foreground">Daily data preprocessing</h3>

      <Stack gap="sm">
        <Switch
          label="Use imputation"
          description="Applies to both classical models and the AI autoencoder."
          checked={useImputation}
          onChange={(event) => onUseImputationChange(event.currentTarget.checked)}
          size="sm"
        />

        {useImputation ? (
          <>
            <Select
              label={
                <span className="inline-flex items-center gap-1">
                  Imputation method
                  <Tooltip
                    label="How to fill missing daily milk values before both model families run"
                    withArrow
                    multiline
                    w={250}
                  >
                    <Info size={14} className="cursor-help text-muted-foreground" />
                  </Tooltip>
                </span>
              }
              data={IMPUTATION_OPTIONS.map((option) => ({
                value: option.value,
                label: option.label,
              }))}
              renderOption={({ option }) => {
                const imputationOption = IMPUTATION_OPTIONS.find(
                  (item) => item.value === option.value
                );

                return (
                  <Tooltip
                    label={imputationOption?.description ?? ""}
                    withArrow
                    multiline
                    w={280}
                    position="right"
                  >
                    <div className="flex w-full min-w-0 flex-col">
                      <span className="text-sm font-medium">{option.label}</span>
                      <span className="text-xs text-muted-foreground">
                        {imputationOption?.description}
                      </span>
                    </div>
                  </Tooltip>
                );
              }}
              value={imputationMethod}
              onChange={(value) => {
                if (value) onImputationMethodChange(value as ImputationMethod);
              }}
              size="sm"
            />
            <Text size="xs" c="dimmed">
              {selectedImputationOption.description}
            </Text>
          </>
        ) : (
          <Text size="xs" c="dimmed">
            Missing daily milk values are included as 0 kg for both model families.
          </Text>
        )}
      </Stack>
    </div>
  );
}
