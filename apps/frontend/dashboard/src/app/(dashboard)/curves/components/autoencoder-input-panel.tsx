"use client";

import type { ReactElement } from "react";
import { Select, NumberInput, Collapse, Button, Tooltip, Switch } from "@mantine/core";
import { Info } from "lucide-react";
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

interface AutoencoderInputPanelProps {
  readonly parity: number;
  readonly onParityChange: (parity: number) => void;
  readonly herdId: number | undefined;
  readonly onHerdIdChange: (herdId: number | undefined) => void;
  readonly imputationMethod: ImputationMethod;
  readonly onImputationMethodChange: (method: ImputationMethod) => void;
  readonly herdStats: number[];
  readonly onHerdStatsChange: (stats: number[]) => void;
  readonly onPredict: () => void;
  readonly isLoading: boolean;
}

export function AutoencoderInputPanel({
  parity,
  onParityChange,
  herdId,
  onHerdIdChange,
  imputationMethod,
  onImputationMethodChange,
  herdStats,
  onHerdStatsChange,
  onPredict,
  isLoading,
}: AutoencoderInputPanelProps): ReactElement {
  const [advancedOpened, { toggle: toggleAdvanced }] = useDisclosure(false);
  const [showRaw, setShowRaw] = useState(false);
  const { data: profiles = [] } = useHerdProfiles();
  const profileOptions = [
    { value: "", label: "None (manual)" },
    ...profiles.map((p) => ({ value: String(p.id), label: p.name })),
  ];

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-sm font-medium text-muted-foreground">Autoencoder</h3>

        <div className="space-y-3">
          <Select
            label="Herd profile preset"
            data={profileOptions}
            defaultValue=""
            onChange={(val) => {
              if (!val) return;
              const profile = profiles.find((p) => String(p.id) === val);
              if (profile) onHerdStatsChange(herdProfileToStats(profile));
            }}
            size="sm"
            placeholder="Select a saved profile…"
            clearable
          />

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

          <NumberInput
            label={
              <span className="inline-flex items-center gap-1">
                Herd ID
                <Tooltip
                  label="Farm identifier for looking up herd-level statistics. Leave empty to use global averages."
                  withArrow
                  multiline
                  w={250}
                >
                  <Info size={14} className="cursor-help text-muted-foreground" />
                </Tooltip>
              </span>
            }
            value={herdId ?? ""}
            onChange={(val) => {
              onHerdIdChange(typeof val === "number" ? val : undefined);
            }}
            size="sm"
            placeholder="Optional, e.g. 2942694"
          />

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
            data={IMPUTATION_OPTIONS.map((o) => ({
              value: o.value,
              label: o.label,
            }))}
            value={imputationMethod}
            onChange={(val) => {
              if (val) onImputationMethodChange(val as ImputationMethod);
            }}
            size="sm"
          />
        </div>

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

      {/* Advanced: Herd Statistics */}
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
                : "All values are normalized between 0 and 1. Adjust these to reflect herd-level statistics that inform the autoencoder prediction."}
            </p>
            <HerdStatsForm values={herdStats} onChange={onHerdStatsChange} showRaw={showRaw} />
          </div>
        </Collapse>
      </div>
    </div>
  );
}
