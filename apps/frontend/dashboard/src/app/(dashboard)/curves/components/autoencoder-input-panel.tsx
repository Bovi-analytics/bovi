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
  Stack,
  Switch,
  Text,
  Tooltip,
} from "@mantine/core";
import { AlertCircle, Check, Info } from "lucide-react";
import { useDisclosure } from "@mantine/hooks";
import { useState } from "react";
import { HerdStatsForm } from "@/app/(dashboard)/autoencoder/components/herd-stats-form";
import { useHerdProfiles } from "@/app/(dashboard)/herd-stats/hooks/use-herd-profiles";
import { herdProfileToStats } from "@/lib/herd-profile-utils";

const HERD_STATS_SOURCE_COPY: Record<HerdStatsSourceKind, { label: string; description: string }> =
  {
    dataset: {
      label: "Active dataset",
      description: "Compute herd-level averages from the dataset selected in Data Upload.",
    },
    default: {
      label: "Model default",
      description: "Let the autoencoder use its global training-set average.",
    },
    profile: {
      label: "Saved profile",
      description: "Load a saved herd profile and use its ten aggregate stats.",
    },
    manual: {
      label: "Manual",
      description: "Edit the herd statistics yourself before prediction.",
    },
  };

export type HerdStatsSourceKind = "dataset" | "default" | "profile" | "manual";

interface AutoencoderInputPanelProps {
  readonly parity: number;
  readonly onParityChange: (parity: number) => void;
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

  const sourceOptions: Array<{
    value: HerdStatsSourceKind;
    label: string;
    description: string;
    disabled?: boolean;
  }> = [
    {
      value: "dataset",
      label: datasetLabel ? `Dataset: ${datasetLabel}` : HERD_STATS_SOURCE_COPY.dataset.label,
      description: datasetLabel
        ? HERD_STATS_SOURCE_COPY.dataset.description
        : "Load a dataset in Data Upload to use this option.",
      disabled: !datasetLabel,
    },
    { value: "default", ...HERD_STATS_SOURCE_COPY.default },
    { value: "profile", ...HERD_STATS_SOURCE_COPY.profile },
    { value: "manual", ...HERD_STATS_SOURCE_COPY.manual },
  ];

  const profileOptions = profiles.map((p) => ({ value: String(p.id), label: p.name }));

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-base font-semibold text-foreground">Autoencoder</h3>

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
            <div className="grid gap-2" role="radiogroup" aria-label="Herd stats source">
              {sourceOptions.map((option) => {
                const isSelected = herdStatsSource === option.value;

                return (
                  <button
                    key={option.value}
                    type="button"
                    role="radio"
                    aria-checked={isSelected}
                    disabled={option.disabled}
                    onClick={() => onHerdStatsSourceChange(option.value)}
                    className={[
                      "flex w-full items-start gap-2 rounded-md border px-3 py-2 text-left transition-colors",
                      isSelected
                        ? "border-violet-500 bg-violet-500/10 text-foreground"
                        : "border-border bg-background/40 text-foreground hover:border-violet-400/70 hover:bg-violet-500/5",
                      option.disabled
                        ? "cursor-not-allowed opacity-50 hover:border-border hover:bg-background/40"
                        : "",
                    ].join(" ")}
                  >
                    <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-current">
                      {isSelected && <Check size={11} strokeWidth={3} />}
                    </span>
                    <span className="min-w-0">
                      <span className="block text-sm font-medium leading-5">{option.label}</span>
                      <span className="block text-xs leading-5 text-muted-foreground">
                        {option.description}
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>
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
            {herdStatsSource === "profile" && profileOptions.length === 0 && (
              <Text size="xs" c="dimmed">
                Create a saved profile in Herd Profiles first, or use dataset/default/manual here.
              </Text>
            )}
            {herdStatsSource === "manual" && (
              <Badge size="xs" color="gray" variant="light" w="fit-content">
                Edit values below in &ldquo;Herd Statistics&rdquo;
              </Badge>
            )}
          </Stack>

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
