"use client";

import React from "react";
import { NumberInput, Slider, Tooltip } from "@mantine/core";
import {
  HERD_STATS_METADATA,
  VISIBLE_HERD_STATS_METADATA,
  toNormalized,
  toRaw,
} from "@/data/herd-stats-metadata";

interface HerdStatsFormProps {
  readonly values: number[];
  readonly onChange: (values: number[]) => void;
  readonly showRaw?: boolean;
  readonly showBoth?: boolean;
}

export function HerdStatsForm({
  values,
  onChange,
  showRaw = false,
  showBoth = false,
}: HerdStatsFormProps): React.ReactElement {
  function clampNormalized(value: number) {
    return Math.max(0, Math.min(1, value));
  }

  function clampValue(index: number) {
    const next = [...values];
    next[index] = clampNormalized(next[index] ?? 0);
    onChange(next);
  }

  function handleChange(index: number, value: number) {
    const stat = HERD_STATS_METADATA[index];
    const normalized = showRaw ? toNormalized(stat, value) : value;
    const next = [...values];
    next[index] = normalized;
    onChange(next);
  }

  function handleRawChange(index: number, value: number) {
    const stat = HERD_STATS_METADATA[index];
    const next = [...values];
    next[index] = toNormalized(stat, value);
    onChange(next);
  }

  function handleNormalizedChange(index: number, value: number) {
    const next = [...values];
    next[index] = value;
    onChange(next);
  }

  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
      {VISIBLE_HERD_STATS_METADATA.map((stat) => {
        const normalized = values[stat.index];
        const displayValue = showRaw ? toRaw(stat, normalized) : normalized;
        const sliderMin = showRaw ? stat.rawMin : 0;
        const sliderMax = showRaw ? stat.rawMax : 1;
        const sliderStep = showRaw
          ? Math.max(1, Math.round((stat.rawMax - stat.rawMin) / 100))
          : 0.01;
        const decimalScale = showRaw && stat.unit === "days" ? 0 : 2;
        const rawValue = toRaw(stat, normalized);
        const rawStep = Math.max(1, Math.round((stat.rawMax - stat.rawMin) / 100));
        const rawDecimalScale = stat.unit === "days" ? 0 : 2;

        return (
          <div key={stat.name} className="rounded-md border border-border bg-background/60 p-3">
            <Tooltip label={stat.description} position="top" withArrow multiline w={250}>
              <label className="block cursor-help text-sm font-semibold leading-tight text-foreground">
                {stat.label}
                {showRaw && stat.unit ? (
                  <span className="ml-1 text-xs font-medium text-muted-foreground">
                    ({stat.unit})
                  </span>
                ) : null}
              </label>
            </Tooltip>
            {showBoth ? (
              <div className="mt-2 grid grid-cols-2 gap-2">
                <NumberInput
                  label={stat.unit || "Raw value"}
                  aria-label={`${stat.label} ${stat.unit || "raw value"}`}
                  value={rawValue}
                  onChange={(val) => {
                    if (typeof val === "number") handleRawChange(stat.index, val);
                  }}
                  onBlur={() => clampValue(stat.index)}
                  min={stat.rawMin}
                  max={stat.rawMax}
                  step={rawStep}
                  decimalScale={rawDecimalScale}
                  size="xs"
                />
                <NumberInput
                  label="Normalized 0-1"
                  aria-label={`${stat.label} normalized`}
                  value={normalized}
                  onChange={(val) => {
                    if (typeof val === "number") handleNormalizedChange(stat.index, val);
                  }}
                  onBlur={() => clampValue(stat.index)}
                  min={0}
                  max={1}
                  step={0.01}
                  decimalScale={2}
                  size="xs"
                />
              </div>
            ) : (
              <div className="mt-3 space-y-2">
                <Slider
                  value={displayValue}
                  onChange={(val) => {
                    const next = [...values];
                    next[stat.index] = showRaw
                      ? clampNormalized(toNormalized(stat, val))
                      : clampNormalized(val);
                    onChange(next);
                  }}
                  min={sliderMin}
                  max={sliderMax}
                  step={sliderStep}
                  size="sm"
                  label={(val) =>
                    showRaw && stat.unit === "days" ? Math.round(val).toString() : val.toFixed(2)
                  }
                />
                <NumberInput
                  aria-label={stat.label}
                  value={displayValue}
                  onChange={(val) => {
                    if (typeof val === "number") handleChange(stat.index, val);
                  }}
                  onBlur={() => clampValue(stat.index)}
                  min={sliderMin}
                  max={sliderMax}
                  step={sliderStep}
                  decimalScale={decimalScale}
                  size="xs"
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
