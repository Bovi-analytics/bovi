"use client";

import type { ReactElement } from "react";
import { NumberInput, Slider, Tooltip } from "@mantine/core";
import { HERD_STATS_METADATA, toRaw, toNormalized } from "@/data/herd-stats-metadata";

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
}: HerdStatsFormProps): ReactElement {
  function handleChange(index: number, value: number) {
    const stat = HERD_STATS_METADATA[index];
    const normalized = showRaw ? toNormalized(stat, value) : value;
    const next = [...values];
    next[index] = Math.max(0, Math.min(1, normalized));
    onChange(next);
  }

  function handleRawChange(index: number, value: number) {
    const stat = HERD_STATS_METADATA[index];
    const next = [...values];
    next[index] = Math.max(0, Math.min(1, toNormalized(stat, value)));
    onChange(next);
  }

  function handleNormalizedChange(index: number, value: number) {
    const next = [...values];
    next[index] = Math.max(0, Math.min(1, value));
    onChange(next);
  }

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
      {HERD_STATS_METADATA.map((stat) => {
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
          <div key={stat.name} className="space-y-1">
            <Tooltip label={stat.description} position="top" withArrow multiline w={250}>
              <label className="block min-h-[2.5rem] cursor-help text-xs font-medium leading-tight text-muted-foreground">
                {stat.label}
                {showRaw && stat.unit ? (
                  <span className="ml-1 text-muted-foreground/60">({stat.unit})</span>
                ) : null}
              </label>
            </Tooltip>
            {showBoth ? (
              <div className="grid grid-cols-2 gap-2">
                <NumberInput
                  label={stat.unit || "Raw"}
                  value={rawValue}
                  onChange={(val) => {
                    if (typeof val === "number") handleRawChange(stat.index, val);
                  }}
                  min={stat.rawMin}
                  max={stat.rawMax}
                  step={rawStep}
                  decimalScale={rawDecimalScale}
                  size="xs"
                />
                <NumberInput
                  label="0-1"
                  value={normalized}
                  onChange={(val) => {
                    if (typeof val === "number") handleNormalizedChange(stat.index, val);
                  }}
                  min={0}
                  max={1}
                  step={0.01}
                  decimalScale={2}
                  size="xs"
                />
              </div>
            ) : (
              <>
                <Slider
                  value={displayValue}
                  onChange={(val) => handleChange(stat.index, val)}
                  min={sliderMin}
                  max={sliderMax}
                  step={sliderStep}
                  size="sm"
                  label={(val) =>
                    showRaw && stat.unit === "days" ? Math.round(val).toString() : val.toFixed(2)
                  }
                />
                <NumberInput
                  value={displayValue}
                  onChange={(val) => {
                    if (typeof val === "number") handleChange(stat.index, val);
                  }}
                  min={sliderMin}
                  max={sliderMax}
                  step={sliderStep}
                  decimalScale={decimalScale}
                  size="xs"
                />
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}
