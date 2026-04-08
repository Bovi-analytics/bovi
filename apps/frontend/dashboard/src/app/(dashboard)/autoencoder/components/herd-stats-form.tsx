"use client";

import type { ReactElement } from "react";
import { NumberInput, Slider, Tooltip } from "@mantine/core";
import { HERD_STATS_METADATA } from "@/data/herd-stats-metadata";

interface HerdStatsFormProps {
  readonly values: number[];
  readonly onChange: (values: number[]) => void;
}

export function HerdStatsForm({ values, onChange }: HerdStatsFormProps): ReactElement {
  function handleChange(index: number, value: number) {
    const next = [...values];
    next[index] = Math.max(0, Math.min(1, value));
    onChange(next);
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
      {HERD_STATS_METADATA.map((stat) => (
        <div key={stat.name} className="space-y-1">
          <Tooltip label={stat.description} position="top" withArrow multiline w={250}>
            <label className="block cursor-help text-xs font-medium text-muted-foreground">
              {stat.label}
            </label>
          </Tooltip>
          <Slider
            value={values[stat.index]}
            onChange={(val) => handleChange(stat.index, val)}
            min={0}
            max={1}
            step={0.01}
            size="sm"
            label={(val) => val.toFixed(2)}
          />
          <NumberInput
            value={values[stat.index]}
            onChange={(val) => {
              if (typeof val === "number") handleChange(stat.index, val);
            }}
            min={0}
            max={1}
            step={0.01}
            decimalScale={2}
            size="xs"
          />
        </div>
      ))}
    </div>
  );
}
