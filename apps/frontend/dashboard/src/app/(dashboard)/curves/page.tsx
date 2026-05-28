"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactElement } from "react";
import {
  Button,
  Group,
  Paper,
  Select,
  Stack,
  Text,
  TextInput,
  Tooltip,
  UnstyledButton,
} from "@mantine/core";
import { Dice5 } from "lucide-react";
import Link from "next/link";
import { MODEL_METADATA } from "@/data/model-metadata";
import { EXAMPLE_LACTATIONS, DEFAULT_LACTATION } from "@/data/example-lactations";
import { EXAMPLE_AUTOENCODER_DATA } from "@/data/example-autoencoder";
import { DEFAULT_HERD_STATS } from "@/data/herd-stats-metadata";
import { preparePeriodicModelInput } from "@/lib/daily-model-input";
import { LactationCurveChart } from "@/components/charts/lactation-curve-chart";
import { ClassicalInputPanel } from "./components/classical-input-panel";
import { AutoencoderInputPanel } from "./components/autoencoder-input-panel";
import type { HerdStatsSourceKind } from "./components/autoencoder-input-panel";
import { StatsComparisonTable } from "./components/stats-comparison-table";
import type { StatsRow } from "./components/stats-comparison-table";
import { useComparison } from "./hooks/use-comparison";
import { useAllCharacteristics } from "./hooks/use-all-characteristics";
import { useAutoencoderPredict } from "./hooks/use-autoencoder-predict";
import { usePresetDataset } from "./hooks/use-preset-dataset";
import { usePresetHerdStats } from "./hooks/use-preset-herd-stats";
import type { Model, MilkBotRunOptions, PresetCow, PresetDatasetKey } from "@/types/api";
import type { ExampleLactation } from "@/data/example-lactations";
import { useWeightUnit } from "@/app/providers/unit-provider";
import { useUploadedCows, type UploadedCow } from "@/app/providers/uploaded-cows-provider";
import {
  ActiveDatasetPanel,
  useActiveDatasetLabel,
} from "@/components/dashboard/active-dataset-panel";

const AUTOENCODER_COLOR = "#ec4899";
const AUTOENCODER_LABEL = "AI autoencoder";
const UPLOADED_COW_ID_PREFIX = "uploaded-";
const DEFAULT_UPLOADED_PARITY = 3;
const UPLOADED_COWS_VISIBLE_CAP = 20;
const PRESET_COWS_VISIBLE_CAP = 50;

const PRESET_DATASET_LABELS: Record<PresetDatasetKey, string> = {
  aurora: "Demo herd A",
  sunnyside: "Demo herd B",
};
const PRESET_PERIOD_LABELS = {
  recent: "Recent",
  old: "Old",
  mixed: "Mixed",
};

function uploadedCowToLactation(cow: UploadedCow, datasetName: string): ExampleLactation {
  return {
    id: `${UPLOADED_COW_ID_PREFIX}${cow.cowId}`,
    label: `${datasetName} - lactation ${cow.cowId}${
      cow.parity != null ? ` (parity ${cow.parity})` : ""
    }`,
    description: `Uploaded test-day records from one lactation in ${datasetName}.${
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
    label: `Lactation ${cow.cow_id}`,
    description: `Anonymized demo-herd test-day records from ${PRESET_DATASET_LABELS[dataset]}.`,
    parity: cow.parity ?? DEFAULT_UPLOADED_PARITY,
    breed: "H",
    dim: [...cow.dim],
    milkrecordings: [...cow.milk_kg],
    source: "icar",
  };
}

function autoencoderExampleToLactation(
  example: (typeof EXAMPLE_AUTOENCODER_DATA)[number]
): ExampleLactation {
  const observed = example.milk
    .map((value, index) => (value !== null ? { dim: index + 1, milk: value } : null))
    .filter((value): value is { dim: number; milk: number } => value !== null);

  return {
    id: `daily-example:${example.id}`,
    label: example.label,
    description: example.description,
    parity: example.parity,
    breed: "H",
    dim: observed.map((value) => value.dim),
    milkrecordings: observed.map((value) => value.milk),
    source: "synthetic",
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
  readonly milkbotOptions: MilkBotRunOptions;
}

function useClassicalResults({
  models,
  dim,
  milkrecordings,
  parity,
  milkbotOptions,
}: ClassicalResultsProps) {
  const fitResults = useComparison({ models, dim, milkrecordings, parity, milkbotOptions });

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

  const allChars = useAllCharacteristics({ models, dim, milkrecordings, parity, milkbotOptions });

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
      name: AUTOENCODER_LABEL,
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
    name: AUTOENCODER_LABEL,
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
  const activeDatasetLabel = useActiveDatasetLabel();

  // Cow source for unified curve records
  const [cowSource, setCowSource] = useState<"data" | "example">("data");

  const [activeLactation, setActiveLactation] = useState<ExampleLactation>(DEFAULT_LACTATION);
  const [cowIdInput, setCowIdInput] = useState("");
  const [cowIdError, setCowIdError] = useState<string | null>(null);

  // Active preset cows - fetched from blob via React Query (shared cache with Data Upload tab)
  const { data: presetData } = usePresetDataset(
    activePreset?.dataset ?? null,
    activePreset?.size ?? "small",
    activePreset?.period ?? "mixed"
  );

  const [parity, setParity] = useState<number>(DEFAULT_LACTATION.parity);
  const [herdStatsSource, setHerdStatsSource] = useState<HerdStatsSourceKind>(
    activePreset ? "dataset" : "default"
  );
  const [selectedProfileId, setSelectedProfileId] = useState<number | null>(null);
  const [manualHerdStats, setManualHerdStats] = useState<number[]>([...DEFAULT_HERD_STATS]);
  const [predictEnabled, setPredictEnabled] = useState(false);

  // Fetch herd stats computed from the active demo herd (only when needed).
  const {
    statsArray: datasetHerdStats,
    isLoading: datasetHerdStatsLoading,
    isError: datasetHerdStatsError,
  } = usePresetHerdStats(
    herdStatsSource === "dataset" ? (activePreset?.dataset ?? null) : null,
    activePreset?.size ?? "small",
    activePreset?.period ?? "mixed"
  );

  // Auto-flip the source when the active dataset comes or goes
  useEffect(() => {
    if (activePreset && herdStatsSource === "default") {
      setHerdStatsSource("dataset");
    } else if (!activePreset && herdStatsSource === "dataset") {
      setHerdStatsSource("default");
    }
  }, [activePreset, herdStatsSource]);

  useEffect(() => {
    setParity(activeLactation.parity);
    setPredictEnabled(false);
  }, [activeLactation]);

  // Resolve the herd_stats array to send with the predict request.
  // `undefined` → omit the field → autoencoder uses its global fallback.
  const effectiveHerdStats: readonly number[] | undefined = (() => {
    switch (herdStatsSource) {
      case "default":
        return undefined;
      case "dataset":
        return datasetHerdStats;
      case "profile":
        return selectedProfileId !== null ? manualHerdStats : undefined;
      case "manual":
        return manualHerdStats;
    }
  })();

  const datasetLabel = activePreset
    ? `${PRESET_DATASET_LABELS[activePreset.dataset]} · ${activePreset.size} · ${activePreset.period}`
    : null;

  // Classical models state (shared between modes)
  const [selectedModels, setSelectedModels] = useState<Model[]>(["wood"]);
  const [milkbotOptions, setMilkbotOptions] = useState<MilkBotRunOptions>({
    fitting: "frequentist",
    breed: "H",
    continent: "USA",
  });

  const projectedAutoencoderInput = useMemo(
    () => preparePeriodicModelInput(activeLactation.dim, activeLactation.milkrecordings),
    [activeLactation.dim, activeLactation.milkrecordings]
  );

  const classicalDim = activeLactation.dim;
  const classicalMilk = activeLactation.milkrecordings;
  const classicalParity = parity;

  // Classical model results
  const {
    curves: classicalCurves,
    statsRows: classicalStats,
    isLoading: classicalLoading,
  } = useClassicalResults({
    models: selectedModels,
    dim: classicalDim,
    milkrecordings: classicalMilk,
    parity: classicalParity,
    milkbotOptions,
  });

  // Autoencoder prediction (daily mode only)
  const {
    data: autoencoderData,
    isLoading: autoencoderLoading,
    isError: autoencoderError,
    error: autoencoderErrorObj,
  } = useAutoencoderPredict({
    dim: activeLactation.dim,
    milkrecordings: activeLactation.milkrecordings,
    parity,
    herdStats: effectiveHerdStats,
    enabled: predictEnabled && activeLactation.dim.length > 0,
  });

  // Autoencoder curve
  const autoencoderCurves = useMemo(() => {
    if (!autoencoderData) return [];
    return [
      {
        name: AUTOENCODER_LABEL,
        color: AUTOENCODER_COLOR,
        data: autoencoderData.predictions.map((val, i) => ({
          dim: i + 1,
          yield: val,
        })),
      },
    ];
  }, [autoencoderData]);

  // Merge all curves
  const allCurves = [...classicalCurves, ...autoencoderCurves];

  // Observations (scatter points)
  const observations = useMemo(() => {
    return activeLactation.dim.map((d, i) => ({
      dim: d,
      yield: activeLactation.milkrecordings[i],
    }));
  }, [activeLactation]);

  // Merge all stats rows
  const autoencoderStatsRow = computeAutoencoderStats(
    autoencoderData?.predictions,
    autoencoderLoading
  );
  const allStats: StatsRow[] = predictEnabled
    ? [...classicalStats, autoencoderStatsRow]
    : classicalStats;

  const zeroFilledCount = projectedAutoencoderInput.missingCount;

  function handleToggleModel(model: Model) {
    setSelectedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    );
  }

  // The currently-active uploaded lactation ID, if any
  const activeUploadedCowId = activeLactation.id.startsWith(UPLOADED_COW_ID_PREFIX)
    ? activeLactation.id.slice(UPLOADED_COW_ID_PREFIX.length)
    : null;

  // The currently-active demo herd lactation info, if any
  const activePresetInfo = useMemo((): { dataset: PresetDatasetKey; cowId: string } | null => {
    for (const ds of ["aurora", "sunnyside"] as PresetDatasetKey[]) {
      const prefix = `preset:${ds}:`;
      if (activeLactation.id.startsWith(prefix)) {
        return { dataset: ds, cowId: activeLactation.id.slice(prefix.length) };
      }
    }
    return null;
  }, [activeLactation.id]);

  // Options for the "Your data" panel - only the active source (preset takes priority over uploaded CSV)
  const herdSelectData = useMemo(() => {
    const groups: { group: string; items: { value: string; label: string }[] }[] = [];
    if (presetData && activePreset) {
      const prefix = `preset:${activePreset.dataset}:`;
      const visible = [...presetData.cows.slice(0, PRESET_COWS_VISIBLE_CAP)];
      const visibleIds = new Set(visible.map((c) => c.cow_id));
      if (
        activePresetInfo?.dataset === activePreset.dataset &&
        !visibleIds.has(activePresetInfo.cowId)
      ) {
        const active = presetData.cows.find((c) => c.cow_id === activePresetInfo.cowId);
        if (active) {
          visible.push(active);
          visibleIds.add(active.cow_id);
        }
      }
      const items = visible.map((c) => ({
        value: `${prefix}${c.cow_id}`,
        label: `Lactation ${c.cow_id}`,
      }));
      const remaining = presetData.cow_count - visibleIds.size;
      if (remaining > 0)
        items.push({
          value: "__preset_overflow__",
          label: `... and ${remaining} more - use Random or type a lactation ID`,
        });
      const groupLabel = `${PRESET_DATASET_LABELS[activePreset.dataset]} - ${PRESET_PERIOD_LABELS[activePreset.period]} (${presetData.cow_count.toLocaleString()} lactations)`;
      groups.push({ group: groupLabel, items });
    } else if (uploadedDataset && uploadedDataset.cows.length > 0) {
      const visible = [...uploadedDataset.cows.slice(0, UPLOADED_COWS_VISIBLE_CAP)];
      const visibleIds = new Set(visible.map((c) => c.cowId));
      if (activeUploadedCowId && !visibleIds.has(activeUploadedCowId)) {
        const active = uploadedDataset.cows.find((c) => c.cowId === activeUploadedCowId);
        if (active) {
          visible.push(active);
          visibleIds.add(active.cowId);
        }
      }
      const items = visible.map((c) => ({
        value: `${UPLOADED_COW_ID_PREFIX}${c.cowId}`,
        label: `Lactation ${c.cowId}${c.parity != null ? ` (parity ${c.parity})` : ""}`,
      }));
      const remaining = uploadedDataset.cows.length - visibleIds.size;
      if (remaining > 0)
        items.push({
          value: "__overflow__",
          label: `... and ${remaining} more - use Random or type a lactation ID`,
        });
      groups.push({ group: uploadedDataset.name, items });
    }
    return groups;
  }, [uploadedDataset, activeUploadedCowId, presetData, activePreset, activePresetInfo]);

  // Options for the "Examples" panel - built-in synthetic + anonymized reference records
  const exampleSelectData = useMemo(
    () => [
      {
        group: "Synthetic examples",
        items: EXAMPLE_LACTATIONS.filter((l) => l.source === "synthetic").map((l) => ({
          value: l.id,
          label: l.label,
        })),
      },
      {
        group: "Anonymized reference records",
        items: EXAMPLE_LACTATIONS.filter((l) => l.source === "icar").map((l) => ({
          value: l.id,
          label: l.label,
        })),
      },
      {
        group: "Daily-recording examples",
        items: EXAMPLE_AUTOENCODER_DATA.map((l) => ({
          value: `daily-example:${l.id}`,
          label: l.label,
        })),
      },
    ],
    []
  );

  function handleSelectLactation(id: string | null) {
    if (!id || id === "__overflow__" || id === "__preset_overflow__") return;
    // Demo herd lactation
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
    // Uploaded lactation
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
    if (id.startsWith("daily-example:")) {
      const found = EXAMPLE_AUTOENCODER_DATA.find((l) => `daily-example:${l.id}` === id);
      if (found) {
        setActiveLactation(autoencoderExampleToLactation(found));
        setCowIdInput("");
        setCowIdError(null);
      }
      return;
    }

    const found = EXAMPLE_LACTATIONS.find((l) => l.id === id);
    if (found) {
      setActiveLactation(found);
      setCowIdInput("");
      setCowIdError(null);
    }
  }

  function handleRandomCow() {
    // Prefer active demo herd
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
    setCowIdError(`No lactation with ID "${id}" found in loaded datasets.`);
  }

  const hasHerdData =
    (uploadedDataset && uploadedDataset.cows.length > 0) ||
    (presetData && presetData.cows.length > 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Curve Analysis</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Fit, compare, and predict lactation curves using classical models and deep learning.
        </p>
      </div>

      <ActiveDatasetPanel
        emptyText="No dataset loaded."
        actionHref="/data-upload"
        actionLabel={activeDatasetLabel ? "Change" : "Data Upload"}
        showActionWithoutDataset
        compact
      />

      <div className="flex flex-wrap items-center gap-4">
        <button
          onClick={toggleWeightUnit}
          className="rounded-md border border-primary/40 bg-primary/15 px-3 py-1.5 text-sm font-medium text-primary transition-colors hover:bg-primary/25"
        >
          {weightUnit === "kg" ? "kg → lbs" : "lbs → kg"}
        </button>
      </div>

      <div className="rounded-lg border border-border bg-card p-4">
        <Stack gap="sm">
          <div>
            <Text size="md" fw={700}>
              Select a lactation
            </Text>
            <Text size="sm" c="dimmed" mt={4}>
              The same observed lactation record is used for classical curve models and the AI
              autoencoder.
            </Text>
          </div>

          <Group gap="xs">
            {[
              {
                value: "data" as const,
                label: activeDatasetLabel
                  ? `Your data - ${activeDatasetLabel.split(" · ")[0]}`
                  : "Your data",
              },
              { value: "example" as const, label: "Examples" },
            ].map((opt) => (
              <UnstyledButton key={opt.value} onClick={() => setCowSource(opt.value)}>
                <Paper
                  withBorder
                  px="sm"
                  py={6}
                  radius="sm"
                  style={{
                    borderColor:
                      cowSource === opt.value ? "var(--mantine-color-violet-6)" : undefined,
                    borderWidth: cowSource === opt.value ? 2 : 1,
                    cursor: "pointer",
                  }}
                >
                  <Text size="xs" fw={500}>
                    {opt.label}
                  </Text>
                </Paper>
              </UnstyledButton>
            ))}
          </Group>

          {cowSource === "data" &&
            (hasHerdData ? (
              <Group gap="xs" wrap="wrap">
                <Select
                  size="xs"
                  value={activeLactation.id}
                  onChange={handleSelectLactation}
                  data={herdSelectData}
                  allowDeselect={false}
                  searchable
                  placeholder="Pick a lactation..."
                  w={300}
                />
                <Tooltip label="Pick a random lactation from your dataset" withArrow>
                  <Button
                    size="xs"
                    variant="light"
                    leftSection={<Dice5 size={14} />}
                    onClick={handleRandomCow}
                  >
                    Random
                  </Button>
                </Tooltip>
                <TextInput
                  size="xs"
                  placeholder="Lactation ID..."
                  value={cowIdInput}
                  onChange={(e) => {
                    setCowIdInput(e.target.value);
                    if (cowIdError) setCowIdError(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleLoadCowById();
                  }}
                  error={cowIdError}
                  w={130}
                />
                <Button size="xs" variant="subtle" onClick={handleLoadCowById}>
                  Load
                </Button>
              </Group>
            ) : (
              <Text size="xs" c="dimmed">
                No dataset loaded -{" "}
                <Link href="/data-upload" className="underline underline-offset-2">
                  go to Data Upload
                </Link>{" "}
                to pick an anonymized demo herd or upload your own CSV.
              </Text>
            ))}

          {cowSource === "example" && (
            <Select
              size="xs"
              value={activeLactation.id}
              onChange={handleSelectLactation}
              data={exampleSelectData}
              allowDeselect={false}
              searchable
              w={340}
            />
          )}

          <Text size="xs" c="dimmed">
            {activeLactation.dim.length.toLocaleString()} observed record
            {activeLactation.dim.length === 1 ? "" : "s"} selected. The autoencoder projects the
            observations onto a 304-day sequence and zero-fills {zeroFilledCount.toLocaleString()}{" "}
            unobserved day{zeroFilledCount === 1 ? "" : "s"}.
          </Text>
        </Stack>
      </div>

      {/* Main layout: input panel + chart */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Left panel: inputs */}
        <div className="space-y-4 lg:col-span-1">
          <ClassicalInputPanel
            selectedModels={selectedModels}
            onToggleModel={handleToggleModel}
            parity={classicalParity}
            milkbotOptions={milkbotOptions}
            onMilkbotOptionsChange={setMilkbotOptions}
          />

          <AutoencoderInputPanel
            parity={parity}
            onParityChange={setParity}
            herdStatsSource={herdStatsSource}
            onHerdStatsSourceChange={setHerdStatsSource}
            selectedProfileId={selectedProfileId}
            onSelectedProfileIdChange={setSelectedProfileId}
            manualHerdStats={manualHerdStats}
            onManualHerdStatsChange={setManualHerdStats}
            datasetLabel={datasetLabel}
            datasetStatsLoading={datasetHerdStatsLoading}
            datasetStatsError={datasetHerdStatsError}
            onPredict={() => setPredictEnabled(true)}
            isLoading={autoencoderLoading}
          />
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
                Error:{" "}
                {autoencoderErrorObj instanceof Error
                  ? autoencoderErrorObj.message
                  : "Unknown error"}
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
