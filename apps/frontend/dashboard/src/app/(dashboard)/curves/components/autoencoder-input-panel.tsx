"use client";

import type { ReactElement } from "react";
import {
  Alert,
  Badge,
  Button,
  Collapse,
  Loader,
  NumberInput,
  Select,
  SegmentedControl,
  Stack,
  Switch,
  Text,
  Tooltip,
} from "@mantine/core";
import { AlertCircle, Info } from "lucide-react";
import { useDisclosure } from "@mantine/hooks";
import { useState } from "react";
import { HerdStatsForm } from "@/app/(dashboard)/autoencoder/components/herd-stats-form";
import { useHerdProfiles } from "@/app/(dashboard)/herd-stats/hooks/use-herd-profiles";
import { herdProfileToStats } from "@/lib/herd-profile-utils";
import type { ImputationMethod } from "@/types/api";

const IMPUTATION_OPTIONS = [
  { value: "forward_fill", label: "Forward fill" },
  { value: "backward_fill", label: "Backward fill" },
  { value: "linear", label: "Linear interpolation" },
  { value: "zero", label: "Zero" },
  { value: "mean", label: "Mean" },
] as const;

export type HerdStatsSourceKind = "dataset" | "default" | "profile" | "manual";

interface AutoencoderInputPanelProps {
  readonly parity: number;
  readonly onParityChange: (parity: number) => void;
  readonly imputationMethod: ImputationMethod;
  readonly onImputationMethodChange: (method: ImputationMethod) => void;
  readonly herdStatsSource: HerdStatsSourceKind;
  readonly onHerdStatsSourceChange: (source: HerdStatsSourceKind) => void;
  readonly selectedProfileId: number | null;
  readonly onSelectedProfileIdChange: (id: number | null) => void;
  readonly manualHerdStats: number[];
  readonly onManualHerdStatsChange: (stats: number[]) => void;
  readonly datasetLabel: string | null;
  readonly datasetStatsLoading: boolean;
  readonly datasetStatsError: boolean;
  readonly onPredict: () => void;
  readonly isLoading: boolean;
}

export function AutoencoderInputPanel({
  parity,
  onParityChange,
  imputationMethod,
  onImputationMethodChange,
  herdStatsSource,
  onHerdStatsSourceChange,
  selectedProfileId,
  onSelectedProfileIdChange,
  manualHerdStats,
  onManualHerdStatsChange,
  datasetLabel,
  datasetStatsLoading,
  datasetStatsError,
  onPredict,
  isLoading,
}: AutoencoderInputPanelProps): ReactElement {
  const [advancedOpened, { toggle: toggleAdvanced }] = useDisclosure(false);
  const [showRaw, setShowRaw] = useState(false);
  const { data: profiles = [] } = useHerdProfiles();

  const sourceOptions = [
    {
      value: "dataset",
      label: datasetLabel ? `Dataset (${datasetLabel})` : "Dataset",
      disabled: !datasetLabel,
    },
    { value: "default", label: "Model default" },
    { value: "profile", label: "Saved profile" },
    { value: "manual", label: "Manual" },
  ];

  const profileOptions = profiles.map((p) => ({ value: String(p.id), label: p.name }));

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-sm font-medium text-muted-foreground">Autoencoder</h3>

        <Stack gap="sm">
          <NumberInput
            label={
              <span className="inline-flex items-center gap-1">
                Parity
                <Tooltip
                  label="Lactation number: 1 = first lactation (heifer), 2+ = multiparous cow"
                  withArrow
                  multiline
                  w={250}
                >
                  <Info size={14} className="cursor-help text-muted-foreground" />
                </Tooltip>
              </span>
            }
            value={parity}
            onChange={(val) => {
              if (typeof val === "number") onParityChange(val);
            }}
            min={1}
            max={12}
            clampBehavior="strict"
            allowDecimal={false}
            size="sm"
          />

          <Stack gap={4}>
            <Text size="sm" fw={500}>
              <span className="inline-flex items-center gap-1">
                Herd stats
                <Tooltip
                  label="Herd-level context for the autoencoder. Pick the active dataset to compute stats from it, fall back to the model's training-set average, load a saved profile, or edit values manually."
                  withArrow
                  multiline
                  w={280}
                >
                  <Info size={14} className="cursor-help text-muted-foreground" />
                </Tooltip>
              </span>
            </Text>
            <SegmentedControl
              size="xs"
              value={herdStatsSource}
              onChange={(v) => onHerdStatsSourceChange(v as HerdStatsSourceKind)}
              data={sourceOptions}
            />
            {herdStatsSource === "dataset" && datasetLabel && (
              <Text size="xs" c="dimmed">
                Computing herd-level averages from <b>{datasetLabel}</b>{" "}
                {datasetStatsLoading && <Loader size="xs" ml={4} />}
              </Text>
            )}
            {herdStatsSource === "dataset" && datasetStatsError && (
              <Alert icon={<AlertCircle size={14} />} color="red" p="xs">
                Could not compute stats from the dataset.
              </Alert>
            )}
            {herdStatsSource === "default" && (
              <Text size="xs" c="dimmed">
                The autoencoder will use its global training-set average.
              </Text>
            )}
            {herdStatsSource === "profile" && (
              <Select
                data={profileOptions}
                value={selectedProfileId !== null ? String(selectedProfileId) : null}
                onChange={(val) => {
                  if (!val) {
                    onSelectedProfileIdChange(null);
                    return;
                  }
                  const profile = profiles.find((p) => String(p.id) === val);
                  if (profile) {
                    onSelectedProfileIdChange(profile.id);
                    onManualHerdStatsChange(herdProfileToStats(profile));
                  }
                }}
                size="xs"
                placeholder={
                  profileOptions.length === 0 ? "No saved profiles yet" : "Select a profile…"
                }
                disabled={profileOptions.length === 0}
              />
            )}
            {herdStatsSource === "manual" && (
              <Badge size="xs" color="gray" variant="light" w="fit-content">
                Edit values below in &ldquo;Herd Statistics&rdquo;
              </Badge>
            )}
          </Stack>

          <Select
            label={
              <span className="inline-flex items-center gap-1">
                Imputation method
                <Tooltip
                  label="How to fill missing (null) values in the milk sequence before prediction"
                  withArrow
                  multiline
                  w={250}
                >
                  <Info size={14} className="cursor-help text-muted-foreground" />
                </Tooltip>
              </span>
            }
            data={IMPUTATION_OPTIONS.map((o) => ({ value: o.value, label: o.label }))}
            value={imputationMethod}
            onChange={(val) => {
              if (val) onImputationMethodChange(val as ImputationMethod);
            }}
            size="sm"
          />
        </Stack>

        <Button
          onClick={onPredict}
          fullWidth
          size="sm"
          color="violet"
          className="mt-4"
          loading={isLoading}
        >
          Predict
        </Button>
      </div>

      {/* Advanced: Herd Statistics — only meaningful when source = manual */}
      {herdStatsSource === "manual" && (
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <Button variant="subtle" size="sm" onClick={toggleAdvanced}>
              {advancedOpened ? "Hide" : "Show"} Herd Statistics
            </Button>
            {advancedOpened && (
              <Switch
                label="Raw values"
                size="xs"
                checked={showRaw}
                onChange={(e) => setShowRaw(e.currentTarget.checked)}
              />
            )}
          </div>
          <Collapse in={advancedOpened}>
            <div className="mt-3">
              <p className="mb-4 text-xs text-muted-foreground">
                {showRaw
                  ? "Values shown in original units. These are converted to 0–1 before prediction."
                  : "All values are normalized between 0 and 1. Adjust to reflect herd-level statistics for the autoencoder."}
              </p>
              <HerdStatsForm
                values={manualHerdStats}
                onChange={onManualHerdStatsChange}
                showRaw={showRaw}
              />
            </div>
          </Collapse>
        </div>
      )}
    </div>
  );
}
