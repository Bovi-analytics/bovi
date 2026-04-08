"use client";

import { useState, useMemo } from "react";
import type { ReactElement } from "react";
import { Select, NumberInput, Collapse, Button } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { LactationCurveChart } from "@/components/charts/lactation-curve-chart";
import { StatCard } from "@/app/(dashboard)/models/components/stat-card";
import {
  EXAMPLE_AUTOENCODER_DATA,
  DEFAULT_AUTOENCODER_DATA,
} from "@/data/example-autoencoder";
import { DEFAULT_HERD_STATS } from "@/data/herd-stats-metadata";
import { HerdStatsForm } from "./components/herd-stats-form";
import { useAutoencoderPredict } from "./hooks/use-autoencoder-predict";
import type { ExampleAutoencoderData } from "@/data/example-autoencoder";
import type { ImputationMethod } from "@/types/api";

/* ------------------------------------------------------------------ */
/*  Imputation method options                                          */
/* ------------------------------------------------------------------ */

const IMPUTATION_OPTIONS = [
  { value: "forward_fill", label: "Forward fill" },
  { value: "backward_fill", label: "Backward fill" },
  { value: "linear", label: "Linear interpolation" },
  { value: "zero", label: "Zero" },
  { value: "mean", label: "Mean" },
] as const;

/* ------------------------------------------------------------------ */
/*  Statistics derived from predictions                                */
/* ------------------------------------------------------------------ */

interface PredictionStat {
  readonly name: string;
  readonly value: number | null;
  readonly isLoading: boolean;
}

function computeStats(
  predictions: readonly number[] | undefined,
  isLoading: boolean
): PredictionStat[] {
  if (!predictions || predictions.length === 0) {
    return [
      { name: "peak_yield", value: null, isLoading },
      { name: "time_to_peak", value: null, isLoading },
      { name: "cumulative_milk_yield", value: null, isLoading },
      { name: "persistency", value: null, isLoading },
    ];
  }

  const peakYield = Math.max(...predictions);
  const timeToPeak = predictions.indexOf(peakYield) + 1;
  const cumulativeYield = predictions.reduce((sum, v) => sum + v, 0);
  const yieldAt200 = predictions[199] ?? predictions[predictions.length - 1];
  const persistency = peakYield > 0 ? yieldAt200 / peakYield : 0;

  return [
    { name: "peak_yield", value: peakYield, isLoading },
    { name: "time_to_peak", value: timeToPeak, isLoading },
    { name: "cumulative_milk_yield", value: cumulativeYield, isLoading },
    { name: "persistency", value: persistency, isLoading },
  ];
}

/* ------------------------------------------------------------------ */
/*  Main page component                                                */
/* ------------------------------------------------------------------ */

export default function AutoencoderPage(): ReactElement {
  const [activeData, setActiveData] = useState<ExampleAutoencoderData>(DEFAULT_AUTOENCODER_DATA);
  const [parity, setParity] = useState<number>(DEFAULT_AUTOENCODER_DATA.parity);
  const [herdId, setHerdId] = useState<number | undefined>(DEFAULT_AUTOENCODER_DATA.herdId);
  const [imputationMethod, setImputationMethod] = useState<ImputationMethod>("forward_fill");
  const [herdStats, setHerdStats] = useState<number[]>([...DEFAULT_HERD_STATS]);
  const [advancedOpened, { toggle: toggleAdvanced }] = useDisclosure(false);
  const [predictEnabled, setPredictEnabled] = useState(false);

  /* Derived: observations for the chart (non-null milk values) */
  const observations = useMemo(
    () =>
      activeData.milk
        .map((val, i) => (val !== null ? { dim: i + 1, yield: val } : null))
        .filter((o): o is NonNullable<typeof o> => o !== null),
    [activeData.milk]
  );

  /* Null count for display */
  const nullCount = activeData.milk.filter((v) => v === null).length;

  /* Prediction query */
  const {
    data: predictionData,
    isLoading,
    isError,
    error,
  } = useAutoencoderPredict({
    milk: activeData.milk,
    parity,
    herdId,
    events: activeData.events,
    herdStats,
    imputationMethod,
    enabled: predictEnabled,
  });

  /* Chart curves from prediction */
  const curves = useMemo(() => {
    if (!predictionData) return [];
    return [
      {
        name: "Autoencoder",
        color: "#9333ea",
        data: predictionData.predictions.map((val, i) => ({
          dim: i + 1,
          yield: val,
        })),
      },
    ];
  }, [predictionData]);

  /* Statistics */
  const stats = computeStats(predictionData?.predictions, isLoading);

  function handleSelectExample(id: string) {
    const found = EXAMPLE_AUTOENCODER_DATA.find((d) => d.id === id);
    if (found) {
      setActiveData(found);
      setParity(found.parity);
      setHerdId(found.herdId);
      setPredictEnabled(false);
    }
  }

  function handlePredict() {
    setPredictEnabled(true);
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Autoencoder</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Predict full lactation curves from partial daily milk data using a deep autoencoder.
        </p>
      </div>

      {/* Main layout: input panel + chart panel */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Left panel: inputs */}
        <div className="space-y-4 lg:col-span-1">
          {/* Example dataset selector */}
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="mb-3 text-sm font-medium text-muted-foreground">Input data</h3>

            <Select
              label="Example dataset"
              data={EXAMPLE_AUTOENCODER_DATA.map((d) => ({
                value: d.id,
                label: d.label,
              }))}
              value={activeData.id}
              onChange={(val) => {
                if (val) handleSelectExample(val);
              }}
              size="sm"
            />

            {/* Milk data summary */}
            <div className="mt-3 rounded-md bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">
                {activeData.milk.length} daily records ({nullCount} missing)
              </p>
              <p className="mt-1 font-mono text-xs text-foreground">
                [{activeData.milk.slice(0, 6).map((v) => (v === null ? "null" : v.toFixed(0))).join(", ")}
                , ...]
              </p>
            </div>
          </div>

          {/* Configuration */}
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="mb-3 text-sm font-medium text-muted-foreground">Configuration</h3>

            <div className="space-y-3">
              <NumberInput
                label="Parity"
                value={parity}
                onChange={(val) => {
                  if (typeof val === "number") setParity(val);
                }}
                min={1}
                max={12}
                size="sm"
              />

              <NumberInput
                label="Herd ID (optional)"
                value={herdId ?? ""}
                onChange={(val) => {
                  setHerdId(typeof val === "number" ? val : undefined);
                }}
                size="sm"
                placeholder="e.g. 2942694"
              />

              <Select
                label="Imputation method"
                data={IMPUTATION_OPTIONS.map((o) => ({
                  value: o.value,
                  label: o.label,
                }))}
                value={imputationMethod}
                onChange={(val) => {
                  if (val) setImputationMethod(val as ImputationMethod);
                }}
                size="sm"
              />
            </div>
          </div>

          {/* Predict button */}
          <Button onClick={handlePredict} fullWidth size="md" color="violet">
            Predict
          </Button>
        </div>

        {/* Right panel: chart + stats */}
        <div className="space-y-4 lg:col-span-2">
          {/* Statistics */}
          {predictEnabled && (
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              {stats.map((s) => (
                <StatCard key={s.name} name={s.name} value={s.value} isLoading={isLoading} />
              ))}
            </div>
          )}

          {/* Chart */}
          <div className="rounded-lg border border-border bg-card p-4">
            {isLoading ? (
              <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                Running prediction...
              </div>
            ) : isError ? (
              <div className="flex h-[400px] items-center justify-center text-destructive">
                Error: {error instanceof Error ? error.message : "Unknown error"}
              </div>
            ) : !predictEnabled ? (
              <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                Select a dataset and click Predict to see results.
              </div>
            ) : (
              <LactationCurveChart curves={curves} observations={observations} />
            )}
          </div>
        </div>
      </div>

      {/* Advanced: Herd Statistics */}
      <div className="rounded-lg border border-border bg-card p-4">
        <Button
          variant="subtle"
          size="sm"
          onClick={toggleAdvanced}
          className="mb-2"
        >
          {advancedOpened ? "Hide" : "Show"} Advanced: Herd Statistics
        </Button>
        <Collapse in={advancedOpened}>
          <div className="mt-3">
            <p className="mb-4 text-xs text-muted-foreground">
              All values are normalized between 0 and 1. Adjust these to reflect herd-level
              statistics that inform the autoencoder prediction.
            </p>
            <HerdStatsForm values={herdStats} onChange={setHerdStats} />
          </div>
        </Collapse>
      </div>
    </div>
  );
}
