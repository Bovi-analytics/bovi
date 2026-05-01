"use client";

import { useState, useMemo } from "react";
import type { ReactElement } from "react";
import { Badge, Button, Group, Paper, SegmentedControl, Select, Stack, Text, TextInput, Tooltip, UnstyledButton } from "@mantine/core";
import { Database, Dice5, Info } from "lucide-react";
import Link from "next/link";
import { MODEL_METADATA } from "@/data/model-metadata";
import { EXAMPLE_LACTATIONS, DEFAULT_LACTATION } from "@/data/example-lactations";
import {
  EXAMPLE_AUTOENCODER_DATA,
  DEFAULT_AUTOENCODER_DATA,
} from "@/data/example-autoencoder";
import { DEFAULT_HERD_STATS } from "@/data/herd-stats-metadata";
import { sampleTestDays } from "@/lib/sample-test-days";
import { LactationCurveChart } from "@/components/charts/lactation-curve-chart";
import { ClassicalInputPanel } from "./components/classical-input-panel";
import { AutoencoderInputPanel } from "./components/autoencoder-input-panel";
import { StatsComparisonTable } from "./components/stats-comparison-table";
import type { StatsRow } from "./components/stats-comparison-table";
import { useComparison } from "./hooks/use-comparison";
import { useAllCharacteristics } from "./hooks/use-all-characteristics";
import { useAutoencoderPredict } from "./hooks/use-autoencoder-predict";
import { usePresetDataset } from "./hooks/use-preset-dataset";
import type { Model, ImputationMethod, PresetCow, PresetDatasetKey } from "@/types/api";
import type { ExampleLactation } from "@/data/example-lactations";
import type { ExampleAutoencoderData } from "@/data/example-autoencoder";
import { useWeightUnit } from "@/app/providers/unit-provider";
import {
  useUploadedCows,
  type UploadedCow,
} from "@/app/providers/uploaded-cows-provider";

const AUTOENCODER_COLOR = "#ec4899";
const UPLOADED_COW_ID_PREFIX = "uploaded-";
const DEFAULT_UPLOADED_PARITY = 3;
const UPLOADED_COWS_VISIBLE_CAP = 20;
const PRESET_COWS_VISIBLE_CAP = 50;

const PRESET_DATASET_LABELS: Record<PresetDatasetKey, string> = {
  aurora: "Aurora Ridge",
  sunnyside: "Sunnyside",
};
const PRESET_PERIOD_LABELS = {
  recent: "Recent",
  old: "Old",
  mixed: "Mixed",
};

type DataMode = "testday" | "daily";

function uploadedCowToLactation(cow: UploadedCow, datasetName: string): ExampleLactation {
  return {
    id: `${UPLOADED_COW_ID_PREFIX}${cow.cowId}`,
    label: `${datasetName} — cow ${cow.cowId}${
      cow.parity != null ? ` (parity ${cow.parity})` : ""
    }`,
    description: `Uploaded test-day records from ${datasetName}.${
      cow.parity == null
        ? ` Parity not available in the source file; defaulted to ${DEFAULT_UPLOADED_PARITY}.`
        : ""
    }`,
    parity: cow.parity ?? DEFAULT_UPLOADED_PARITY,
    breed: "H",
    dim: [...cow.dim],
    milkrecordings: [...cow.milkrecordings],
    source: "icar",
  };
}

function presetCowToLactation(cow: PresetCow, dataset: PresetDatasetKey): ExampleLactation {
  return {
    id: `preset:${dataset}:${cow.cow_id}`,
    label: cow.display_name,
    description: `${PRESET_DATASET_LABELS[dataset]} test-day records.`,
    parity: cow.parity ?? DEFAULT_UPLOADED_PARITY,
    breed: "H",
    dim: [...cow.dim],
    milkrecordings: [...cow.milk_kg],
    source: "icar",
  };
}

/* ------------------------------------------------------------------ */
/*  Classical models chart + stats (shared between both modes)         */
/* ------------------------------------------------------------------ */

interface ClassicalResultsProps {
  readonly models: readonly Model[];
  readonly dim: readonly number[];
  readonly milkrecordings: readonly number[];
  readonly parity: number;
}

function useClassicalResults({ models, dim, milkrecordings, parity }: ClassicalResultsProps) {
  const fitResults = useComparison({ models, dim, milkrecordings });

  const curves = fitResults
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

  const isLoading = fitResults.some((r) => r.isLoading);

  const allChars = useAllCharacteristics({ models, dim, milkrecordings, parity });

  const statsRows: StatsRow[] = allChars.map((chars) => {
    const metadata = MODEL_METADATA[chars.model];
    return {
      name: metadata.name,
      color: metadata.color,
      peakYield: chars.peakYield,
      timeToPeak: chars.timeToPeak,
      cumulativeYield: chars.cumulativeYield,
      persistency: chars.persistency,
      isLoading: chars.isLoading,
    };
  });

  return { curves, statsRows, isLoading };
}

/* ------------------------------------------------------------------ */
/*  Autoencoder stats helper                                           */
/* ------------------------------------------------------------------ */

function computeAutoencoderStats(
  predictions: readonly number[] | undefined,
  isLoading: boolean
): StatsRow {
  if (!predictions || predictions.length === 0) {
    return {
      name: "Autoencoder",
      color: AUTOENCODER_COLOR,
      peakYield: null,
      timeToPeak: null,
      cumulativeYield: null,
      persistency: null,
      isLoading,
    };
  }

  const peakYield = Math.max(...predictions);
  const timeToPeak = predictions.indexOf(peakYield) + 1;
  const cumulativeYield = predictions.reduce((sum, v) => sum + v, 0);
  const yieldAt200 = predictions[199] ?? predictions[predictions.length - 1];
  const persistency = peakYield > 0 ? yieldAt200 / peakYield : 0;

  return {
    name: "Autoencoder",
    color: AUTOENCODER_COLOR,
    peakYield,
    timeToPeak,
    cumulativeYield,
    persistency,
    isLoading,
  };
}

/* ------------------------------------------------------------------ */
/*  Main page component                                                */
/* ------------------------------------------------------------------ */

export default function CurvesPage(): ReactElement {
  const { weightUnit, toggleWeightUnit } = useWeightUnit();
  const { dataset: uploadedDataset, getCow, getRandomCow, activePreset } = useUploadedCows();

  // Data mode
  const [dataMode, setDataMode] = useState<DataMode>("testday");

  // Cow source for periodic records mode
  const [cowSource, setCowSource] = useState<"data" | "example">("data");
  // Cow source for daily recordings mode
  const [cowSourceDaily, setCowSourceDaily] = useState<"data" | "example">("example");

  // Test-day mode state
  const [activeLactation, setActiveLactation] = useState<ExampleLactation>(DEFAULT_LACTATION);
  const [cowIdInput, setCowIdInput] = useState("");
  const [cowIdError, setCowIdError] = useState<string | null>(null);

  // Active preset cows — fetched from blob via React Query (shared cache with Herd Stats tab)
  const { data: presetData } = usePresetDataset(
    activePreset?.dataset ?? null,
    activePreset?.size ?? "small",
    activePreset?.period ?? "mixed"
  );

  // Daily recordings mode state
  const [activeAutoData, setActiveAutoData] = useState<ExampleAutoencoderData>(
    DEFAULT_AUTOENCODER_DATA
  );
  const [parity, setParity] = useState<number>(DEFAULT_AUTOENCODER_DATA.parity);
  const [herdId, setHerdId] = useState<number | undefined>(DEFAULT_AUTOENCODER_DATA.herdId);
  const [imputationMethod, setImputationMethod] = useState<ImputationMethod>("forward_fill");
  const [herdStats, setHerdStats] = useState<number[]>([...DEFAULT_HERD_STATS]);
  const [predictEnabled, setPredictEnabled] = useState(false);

  // Classical models state (shared between modes)
  const [selectedModels, setSelectedModels] = useState<Model[]>(["wood"]);

  // Derive sparse test-day points from dense data (daily mode)
  const sampledTestDays = useMemo(
    () => sampleTestDays(activeAutoData.milk),
    [activeAutoData.milk]
  );

  // Choose dim/milkrecordings based on mode
  const classicalDim = dataMode === "testday" ? activeLactation.dim : sampledTestDays.dim;
  const classicalMilk =
    dataMode === "testday" ? activeLactation.milkrecordings : sampledTestDays.milkrecordings;
  const classicalParity = dataMode === "testday" ? activeLactation.parity : parity;

  // Classical model results
  const { curves: classicalCurves, statsRows: classicalStats, isLoading: classicalLoading } =
    useClassicalResults({
      models: selectedModels,
      dim: classicalDim,
      milkrecordings: classicalMilk,
      parity: classicalParity,
    });

  // Autoencoder prediction (daily mode only)
  const {
    data: autoencoderData,
    isLoading: autoencoderLoading,
    isError: autoencoderError,
    error: autoencoderErrorObj,
  } = useAutoencoderPredict({
    milk: activeAutoData.milk,
    parity,
    herdId,
    events: activeAutoData.events,
    herdStats,
    imputationMethod,
    enabled: dataMode === "daily" && predictEnabled,
  });

  // Autoencoder curve
  const autoencoderCurves = useMemo(() => {
    if (!autoencoderData || dataMode !== "daily") return [];
    return [
      {
        name: "Autoencoder",
        color: AUTOENCODER_COLOR,
        data: autoencoderData.predictions.map((val, i) => ({
          dim: i + 1,
          yield: val,
        })),
      },
    ];
  }, [autoencoderData, dataMode]);

  // Merge all curves
  const allCurves = [...classicalCurves, ...autoencoderCurves];

  // Observations (scatter points)
  const observations = useMemo(() => {
    if (dataMode === "testday") {
      return activeLactation.dim.map((d, i) => ({
        dim: d,
        yield: activeLactation.milkrecordings[i],
      }));
    }
    return activeAutoData.milk
      .map((val, i) => (val !== null ? { dim: i + 1, yield: val } : null))
      .filter((o): o is NonNullable<typeof o> => o !== null);
  }, [dataMode, activeLactation, activeAutoData.milk]);

  // Merge all stats rows
  const autoencoderStatsRow = computeAutoencoderStats(
    autoencoderData?.predictions,
    autoencoderLoading
  );
  const allStats: StatsRow[] =
    dataMode === "daily" && predictEnabled
      ? [...classicalStats, autoencoderStatsRow]
      : classicalStats;

  // Null count for daily mode display
  const nullCount = activeAutoData.milk.filter((v) => v === null).length;

  function handleToggleModel(model: Model) {
    setSelectedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    );
  }

  function handleSelectAutoExample(id: string) {
    const found = EXAMPLE_AUTOENCODER_DATA.find((d) => d.id === id);
    if (found) {
      setActiveAutoData(found);
      setParity(found.parity);
      setHerdId(found.herdId);
      setPredictEnabled(false);
    }
  }

  // The currently-active uploaded cow ID, if any
  const activeUploadedCowId = activeLactation.id.startsWith(UPLOADED_COW_ID_PREFIX)
    ? activeLactation.id.slice(UPLOADED_COW_ID_PREFIX.length)
    : null;

  // The currently-active preset cow info, if any
  const activePresetInfo = useMemo((): { dataset: PresetDatasetKey; cowId: string } | null => {
    for (const ds of ["aurora", "sunnyside"] as PresetDatasetKey[]) {
      const prefix = `preset:${ds}:`;
      if (activeLactation.id.startsWith(prefix)) {
        return { dataset: ds, cowId: activeLactation.id.slice(prefix.length) };
      }
    }
    return null;
  }, [activeLactation.id]);

  // Options for the "Your data" panel — only the active source (preset takes priority over uploaded CSV)
  const herdSelectData = useMemo(() => {
    const groups: { group: string; items: { value: string; label: string }[] }[] = [];
    if (presetData && activePreset) {
      const prefix = `preset:${activePreset.dataset}:`;
      const visible = [...presetData.cows.slice(0, PRESET_COWS_VISIBLE_CAP)];
      const visibleIds = new Set(visible.map((c) => c.cow_id));
      if (activePresetInfo?.dataset === activePreset.dataset && !visibleIds.has(activePresetInfo.cowId)) {
        const active = presetData.cows.find((c) => c.cow_id === activePresetInfo.cowId);
        if (active) { visible.push(active); visibleIds.add(active.cow_id); }
      }
      const items = visible.map((c) => ({ value: `${prefix}${c.cow_id}`, label: c.display_name }));
      const remaining = presetData.cow_count - visibleIds.size;
      if (remaining > 0) items.push({ value: "__preset_overflow__", label: `… and ${remaining} more — use 🎲 or type a cow ID` });
      const groupLabel = `${PRESET_DATASET_LABELS[activePreset.dataset]} — ${PRESET_PERIOD_LABELS[activePreset.period]} (${presetData.cow_count.toLocaleString()} cows)`;
      groups.push({ group: groupLabel, items });
    } else if (uploadedDataset && uploadedDataset.cows.length > 0) {
      const visible = [...uploadedDataset.cows.slice(0, UPLOADED_COWS_VISIBLE_CAP)];
      const visibleIds = new Set(visible.map((c) => c.cowId));
      if (activeUploadedCowId && !visibleIds.has(activeUploadedCowId)) {
        const active = uploadedDataset.cows.find((c) => c.cowId === activeUploadedCowId);
        if (active) { visible.push(active); visibleIds.add(active.cowId); }
      }
      const items = visible.map((c) => ({
        value: `${UPLOADED_COW_ID_PREFIX}${c.cowId}`,
        label: `Cow ${c.cowId}${c.parity != null ? ` (parity ${c.parity})` : ""}`,
      }));
      const remaining = uploadedDataset.cows.length - visibleIds.size;
      if (remaining > 0) items.push({ value: "__overflow__", label: `… and ${remaining} more — use 🎲 or type a cow ID` });
      groups.push({ group: uploadedDataset.name, items });
    }
    return groups;
  }, [uploadedDataset, activeUploadedCowId, presetData, activePreset, activePresetInfo]);

  // Options for the "Examples" panel — built-in synthetic + ICAR presets
  const exampleSelectData = useMemo(() => [
    {
      group: "Synthetic examples",
      items: EXAMPLE_LACTATIONS.filter((l) => l.source === "synthetic").map((l) => ({ value: l.id, label: l.label })),
    },
    {
      group: "Real ICAR records",
      items: EXAMPLE_LACTATIONS.filter((l) => l.source === "icar").map((l) => ({ value: l.id, label: l.label })),
    },
  ], []);

  function handleSelectLactation(id: string | null) {
    if (!id || id === "__overflow__" || id === "__preset_overflow__") return;
    // Preset cow
    for (const ds of ["aurora", "sunnyside"] as PresetDatasetKey[]) {
      const prefix = `preset:${ds}:`;
      if (id.startsWith(prefix) && presetData) {
        const cowId = id.slice(prefix.length);
        const cow = presetData.cows.find((c) => c.cow_id === cowId);
        if (cow) {
          setActiveLactation(presetCowToLactation(cow, ds));
          setCowIdInput(cow.cow_id);
          setCowIdError(null);
        }
        return;
      }
    }
    // Uploaded cow
    if (id.startsWith(UPLOADED_COW_ID_PREFIX) && uploadedDataset) {
      const cowId = id.slice(UPLOADED_COW_ID_PREFIX.length);
      const cow = getCow(cowId);
      if (cow) {
        setActiveLactation(uploadedCowToLactation(cow, uploadedDataset.name));
        setCowIdInput(cow.cowId);
        setCowIdError(null);
      }
      return;
    }
    // Example lactation
    const found = EXAMPLE_LACTATIONS.find((l) => l.id === id);
    if (found) {
      setActiveLactation(found);
      setCowIdInput("");
      setCowIdError(null);
    }
  }

  function handleRandomCow() {
    // Prefer active preset dataset
    if (presetData && activePreset && presetData.cows.length > 0) {
      const cow = presetData.cows[Math.floor(Math.random() * presetData.cows.length)];
      setActiveLactation(presetCowToLactation(cow, activePreset.dataset));
      setCowIdInput(cow.cow_id);
      setCowIdError(null);
      return;
    }
    // Fall back to uploaded dataset
    if (!uploadedDataset) return;
    const cow = getRandomCow();
    if (cow) {
      setActiveLactation(uploadedCowToLactation(cow, uploadedDataset.name));
      setCowIdInput(cow.cowId);
      setCowIdError(null);
    }
  }

  function handleLoadCowById() {
    const id = cowIdInput.trim();
    if (!id) return;
    // Try preset first
    if (presetData && activePreset) {
      const cow = presetData.cows.find((c) => c.cow_id === id);
      if (cow) {
        setActiveLactation(presetCowToLactation(cow, activePreset.dataset));
        setCowIdInput(cow.cow_id);
        setCowIdError(null);
        return;
      }
    }
    // Try uploaded
    if (uploadedDataset) {
      const cow = getCow(id);
      if (cow) {
        setActiveLactation(uploadedCowToLactation(cow, uploadedDataset.name));
        setCowIdInput(cow.cowId);
        setCowIdError(null);
        return;
      }
    }
    setCowIdError(`No cow with ID "${id}" found in loaded datasets.`);
  }

  const hasHerdData =
    (uploadedDataset && uploadedDataset.cows.length > 0) ||
    (presetData && presetData.cows.length > 0);

  // Human-readable label for the active dataset
  const activeDatasetLabel = (() => {
    if (activePreset && presetData) {
      const name = PRESET_DATASET_LABELS[activePreset.dataset];
      const period = PRESET_PERIOD_LABELS[activePreset.period];
      return `${name} · ${period} · ${presetData.cow_count.toLocaleString()} cows`;
    }
    if (uploadedDataset) {
      return `${uploadedDataset.name} · ${uploadedDataset.cows.length.toLocaleString()} cows`;
    }
    return null;
  })();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Curve Analysis</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Fit, compare, and predict lactation curves using classical models and deep learning.
        </p>
      </div>

      {/* Active dataset indicator */}
      <Group gap="sm" align="center">
        <Database size={14} className="text-muted-foreground" />
        {activeDatasetLabel ? (
          <>
            <Text size="xs" c="dimmed">Dataset:</Text>
            <Badge variant="light" color="violet" size="sm">{activeDatasetLabel}</Badge>
            <Link href="/herd-stats">
              <Text size="xs" c="dimmed" style={{ textDecoration: "underline", textUnderlineOffset: 2 }}>
                Change
              </Text>
            </Link>
          </>
        ) : (
          <>
            <Text size="xs" c="dimmed">No dataset loaded —</Text>
            <Link href="/herd-stats">
              <Text size="xs" c="violet" style={{ textDecoration: "underline", textUnderlineOffset: 2 }}>
                load one in Herd Stats
              </Text>
            </Link>
          </>
        )}
      </Group>

      {/* Data mode toggle + dataset selector on one row */}
      <div className="flex flex-wrap items-center gap-4">
        <SegmentedControl
          value={dataMode}
          onChange={(val) => setDataMode(val as DataMode)}
          data={[
            {
              value: "testday",
              label: (
                <Tooltip
                  label="Sparse measurements taken approximately every 5 weeks — typical milk recording (MPR) data. Used to fit classical parametric models."
                  multiline
                  w={280}
                  withArrow
                  position="bottom"
                >
                  <span className="flex items-center gap-1">
                    Periodic Records
                    <Info size={12} className="text-muted-foreground" />
                  </span>
                </Tooltip>
              ),
            },
            {
              value: "daily",
              label: (
                <Tooltip
                  label="Dense daily measurements, e.g. from a milking robot. Enables the autoencoder to reconstruct the full lactation curve. Classical models also run on sampled points from this data."
                  multiline
                  w={280}
                  withArrow
                  position="bottom"
                >
                  <span className="flex items-center gap-1">
                    Daily Recordings
                    <Info size={12} className="text-muted-foreground" />
                  </span>
                </Tooltip>
              ),
            },
          ]}
        />
        {/* Unit toggle */}
        <button
          onClick={toggleWeightUnit}
          className="rounded-md border border-border bg-card px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted/40 hover:text-foreground"
        >
          {weightUnit === "kg" ? "kg → lbs" : "lbs → kg"}
        </button>
      </div>

      {/* Cow selection (testday mode only) */}
      {dataMode === "testday" && (
        <div className="rounded-lg border border-border bg-card p-4">
          <Stack gap="sm">
            <div>
              <Text size="sm" fw={600}>Select a cow</Text>
              <Text size="xs" c="dimmed" mt={2}>
                Classical models are fit to the test-day records of one individual cow at a time.
              </Text>
            </div>

            {/* Source toggle */}
            <Group gap="xs">
              {[
                { value: "data" as const, label: activeDatasetLabel ? `Your data — ${activeDatasetLabel.split(" · ")[0]}` : "Your data" },
                { value: "example" as const, label: "Examples" },
              ].map((opt) => (
                <UnstyledButton key={opt.value} onClick={() => setCowSource(opt.value)}>
                  <Paper
                    withBorder
                    px="sm"
                    py={6}
                    radius="sm"
                    style={{
                      borderColor: cowSource === opt.value ? "var(--mantine-color-violet-6)" : undefined,
                      borderWidth: cowSource === opt.value ? 2 : 1,
                      cursor: "pointer",
                    }}
                  >
                    <Text size="xs" fw={500}>{opt.label}</Text>
                  </Paper>
                </UnstyledButton>
              ))}
            </Group>

            {/* Your data panel */}
            {cowSource === "data" && (
              hasHerdData ? (
                <Group gap="xs" wrap="wrap">
                  <Select
                    size="xs"
                    value={activeLactation.id}
                    onChange={handleSelectLactation}
                    data={herdSelectData}
                    allowDeselect={false}
                    searchable
                    placeholder="Pick a cow…"
                    w={300}
                  />
                  <Tooltip label="Pick a random cow from your dataset" withArrow>
                    <Button size="xs" variant="light" leftSection={<Dice5 size={14} />} onClick={handleRandomCow}>
                      Random
                    </Button>
                  </Tooltip>
                  <TextInput
                    size="xs"
                    placeholder="Cow ID…"
                    value={cowIdInput}
                    onChange={(e) => { setCowIdInput(e.target.value); if (cowIdError) setCowIdError(null); }}
                    onKeyDown={(e) => { if (e.key === "Enter") handleLoadCowById(); }}
                    error={cowIdError}
                    w={130}
                  />
                  <Button size="xs" variant="subtle" onClick={handleLoadCowById}>Load</Button>
                </Group>
              ) : (
                <Text size="xs" c="dimmed">
                  No dataset loaded —{" "}
                  <Link href="/herd-stats" className="underline underline-offset-2">
                    go to Herd Stats
                  </Link>{" "}
                  to pick Aurora Ridge, Sunnyside, or upload your own CSV.
                </Text>
              )
            )}

            {/* Examples panel */}
            {cowSource === "example" && (
              <Select
                size="xs"
                value={activeLactation.id}
                onChange={handleSelectLactation}
                data={exampleSelectData}
                allowDeselect={false}
                searchable
                w={300}
              />
            )}
          </Stack>
        </div>
      )}

      {/* Cow selection (daily mode) */}
      {dataMode === "daily" && (
        <div className="rounded-lg border border-border bg-card p-4">
          <Stack gap="sm">
            <div>
              <Text size="sm" fw={600}>Select a cow</Text>
              <Text size="xs" c="dimmed" mt={2}>
                Daily recordings enable both classical models and the autoencoder. Preset herd datasets only contain periodic records — use the built-in examples for the autoencoder.
              </Text>
            </div>

            {/* Source toggle */}
            <Group gap="xs">
              {[
                { value: "data" as const, label: activeDatasetLabel ? `Your data — ${activeDatasetLabel.split(" · ")[0]}` : "Your data" },
                { value: "example" as const, label: "Examples" },
              ].map((opt) => (
                <UnstyledButton key={opt.value} onClick={() => setCowSourceDaily(opt.value)}>
                  <Paper
                    withBorder
                    px="sm"
                    py={6}
                    radius="sm"
                    style={{
                      borderColor: cowSourceDaily === opt.value ? "var(--mantine-color-violet-6)" : undefined,
                      borderWidth: cowSourceDaily === opt.value ? 2 : 1,
                      cursor: "pointer",
                    }}
                  >
                    <Text size="xs" fw={500}>{opt.label}</Text>
                  </Paper>
                </UnstyledButton>
              ))}
            </Group>

            {/* Your data panel — preset/uploaded cows have periodic data only */}
            {cowSourceDaily === "data" && (
              activePreset ? (
                <Text size="xs" c="dimmed">
                  {PRESET_DATASET_LABELS[activePreset.dataset]} cows have periodic records, not daily recordings.{" "}
                  Switch to <UnstyledButton onClick={() => setDataMode("testday")} style={{ display: "inline" }}><Text size="xs" c="violet" style={{ display: "inline", textDecoration: "underline", textUnderlineOffset: 2 }}>Periodic Records</Text></UnstyledButton>{" "}
                  to analyze individual cows from your dataset.
                </Text>
              ) : uploadedDataset ? (
                <Text size="xs" c="dimmed">
                  Your uploaded CSV contains periodic records, not daily recordings.{" "}
                  Switch to <UnstyledButton onClick={() => setDataMode("testday")} style={{ display: "inline" }}><Text size="xs" c="violet" style={{ display: "inline", textDecoration: "underline", textUnderlineOffset: 2 }}>Periodic Records</Text></UnstyledButton>{" "}
                  to analyze individual cows from your dataset.
                </Text>
              ) : (
                <Text size="xs" c="dimmed">
                  No dataset loaded —{" "}
                  <Link href="/herd-stats" className="underline underline-offset-2">
                    go to Herd Stats
                  </Link>{" "}
                  to pick Aurora Ridge, Sunnyside, or upload your own CSV.
                </Text>
              )
            )}

            {/* Examples panel */}
            {cowSourceDaily === "example" && (
              <Select
                size="xs"
                value={activeAutoData.id}
                onChange={(id) => { if (id) handleSelectAutoExample(id); }}
                data={EXAMPLE_AUTOENCODER_DATA.map((d) => ({ value: d.id, label: d.label }))}
                allowDeselect={false}
                searchable
                w={340}
              />
            )}

            {/* Data summary — always visible in daily mode */}
            <Text size="xs" c="dimmed">
              {activeAutoData.milk.length} daily records ({nullCount} missing) · autoencoder uses the full sequence · classical models use {sampledTestDays.dim.length} sampled test-day points
            </Text>
          </Stack>
        </div>
      )}

      {/* Main layout: input panel + chart */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Left panel: inputs */}
        <div className="space-y-4 lg:col-span-1">
          <ClassicalInputPanel
            selectedModels={selectedModels}
            onToggleModel={handleToggleModel}
          />

          {dataMode === "daily" && (
            <AutoencoderInputPanel
              parity={parity}
              onParityChange={setParity}
              herdId={herdId}
              onHerdIdChange={setHerdId}
              imputationMethod={imputationMethod}
              onImputationMethodChange={setImputationMethod}
              herdStats={herdStats}
              onHerdStatsChange={setHerdStats}
              onPredict={() => setPredictEnabled(true)}
              isLoading={autoencoderLoading}
            />
          )}
        </div>

        {/* Right panel: chart + stats */}
        <div className="space-y-4 lg:col-span-2">
          {/* Chart */}
          <div className="rounded-lg border border-border bg-card p-4">
            {classicalLoading && allCurves.length === 0 ? (
              <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                Fitting models...
              </div>
            ) : autoencoderError ? (
              <div className="flex h-[400px] items-center justify-center text-destructive">
                Error: {autoencoderErrorObj instanceof Error ? autoencoderErrorObj.message : "Unknown error"}
              </div>
            ) : allCurves.length === 0 && selectedModels.length === 0 ? (
              <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                Select at least one model to see results.
              </div>
            ) : (
              <LactationCurveChart curves={allCurves} observations={observations} />
            )}
          </div>

          {/* Statistics comparison table */}
          <StatsComparisonTable rows={allStats} />
        </div>
      </div>
    </div>
  );
}
