"use client";

import { useState } from "react";
import type { ReactElement } from "react";
import { Checkbox } from "@mantine/core";
import { ALL_MODELS, MODEL_METADATA } from "@/data/model-metadata";
import { EXAMPLE_LACTATIONS, DEFAULT_LACTATION } from "@/data/example-lactations";
import type { Model } from "@/types/api";
import type { ExampleLactation } from "@/data/example-lactations";
import { useComparison } from "./hooks/use-comparison";
import { useCharacteristics } from "./hooks/use-characteristics";
import { ModelInfo } from "./components/model-info";
import { StatCard } from "./components/stat-card";
import { LactationCurveChart } from "@/components/charts/lactation-curve-chart";

/* ------------------------------------------------------------------ */
/*  Chart section — fits all selected models in parallel               */
/* ------------------------------------------------------------------ */

interface ModelChartProps {
  readonly models: readonly Model[];
  readonly lactation: ExampleLactation;
}

function ModelChart({ models, lactation }: ModelChartProps): ReactElement {
  const results = useComparison({
    models,
    dim: lactation.dim,
    milkrecordings: lactation.milkrecordings,
  });

  const isLoading = results.some((r) => r.isLoading);

  const observations = lactation.dim.map((d, i) => ({
    dim: d,
    yield: lactation.milkrecordings[i],
  }));

  const curves = results
    .map((result, i) => {
      if (!result.data) return null;
      const metadata = MODEL_METADATA[models[i]];
      return {
        name: metadata.name,
        color: metadata.color,
        data: result.data.predictions.map((val: number, j: number) => ({
          dim: j + 1,
          yield: val,
        })),
      };
    })
    .filter((c): c is NonNullable<typeof c> => c !== null);

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {isLoading ? (
        <div className="flex h-[400px] items-center justify-center text-muted-foreground">
          Fitting models…
        </div>
      ) : (
        <LactationCurveChart curves={curves} observations={observations} />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Characteristics section — stat cards for one model                 */
/* ------------------------------------------------------------------ */

interface ModelCharacteristicsProps {
  readonly model: Model;
  readonly lactation: ExampleLactation;
}

function ModelCharacteristics({ model, lactation }: ModelCharacteristicsProps): ReactElement {
  const metadata = MODEL_METADATA[model];
  const { characteristics } = useCharacteristics({
    model,
    dim: lactation.dim,
    milkrecordings: lactation.milkrecordings,
    parity: lactation.parity,
  });

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-medium text-muted-foreground">
        Characteristics — {metadata.name}
      </h3>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {characteristics.map((c) => (
          <StatCard key={c.name} name={c.name} value={c.value} isLoading={c.isLoading} />
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main page component                                                */
/* ------------------------------------------------------------------ */

export default function ModelsPage(): ReactElement {
  const [selectedModels, setSelectedModels] = useState<Model[]>(["wood"]);
  const [activeLactation, setActiveLactation] = useState<ExampleLactation>(DEFAULT_LACTATION);

  function handleToggleModel(model: Model) {
    setSelectedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Models</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Fit, compare, and explore lactation curve models.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-6">
        {/* Dataset selector */}
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-muted-foreground">Dataset:</label>
          <select
            value={activeLactation.id}
            onChange={(e) => {
              const found = EXAMPLE_LACTATIONS.find((l) => l.id === e.target.value);
              if (found) setActiveLactation(found);
            }}
            className="rounded-md border border-border bg-card px-3 py-1.5 text-sm text-foreground"
          >
            {EXAMPLE_LACTATIONS.map((lac) => (
              <option key={lac.id} value={lac.id}>
                {lac.label}
              </option>
            ))}
          </select>
        </div>

        {/* Model checkboxes */}
        <div className="flex items-center gap-4">
          {ALL_MODELS.map((m) => (
            <Checkbox
              key={m.id}
              label={m.name}
              checked={selectedModels.includes(m.id)}
              onChange={() => handleToggleModel(m.id)}
              color={m.color}
              size="sm"
            />
          ))}
        </div>
      </div>

      {/* Characteristics for each selected model */}
      {selectedModels.map((model) => (
        <ModelCharacteristics key={model} model={model} lactation={activeLactation} />
      ))}

      {/* Chart */}
      {selectedModels.length > 0 ? (
        <ModelChart models={selectedModels} lactation={activeLactation} />
      ) : (
        <div className="rounded-lg border border-border bg-card p-8 text-center text-muted-foreground">
          Select at least one model to compare.
        </div>
      )}

      {/* Model info for each selected model */}
      {selectedModels.map((model) => (
        <ModelInfo key={model} model={MODEL_METADATA[model]} />
      ))}
    </div>
  );
}
