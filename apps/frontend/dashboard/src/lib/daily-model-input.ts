export const DAILY_MODEL_INPUT_DAYS = 304;

export type DailyImputationMethod = "forward_fill" | "backward_fill" | "linear";

interface DailyModelInputOptions {
  readonly useImputation: boolean;
  readonly imputationMethod: DailyImputationMethod;
  readonly maxDays?: number;
}

interface DailyModelInput {
  readonly dim: number[];
  readonly milk: number[];
  readonly missingCount: number;
}

function isObservedMilk(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function zeroFill(values: readonly (number | null | undefined)[]): number[] {
  return values.map((value) => (isObservedMilk(value) ? value : 0));
}

function getObservedIndexes(values: readonly (number | null | undefined)[]): number[] {
  return values.reduce<number[]>((indexes, value, index) => {
    if (isObservedMilk(value)) indexes.push(index);
    return indexes;
  }, []);
}

function forwardFill(values: readonly (number | null | undefined)[]): number[] {
  const observedIndexes = getObservedIndexes(values);
  if (observedIndexes.length === 0) return zeroFill(values);

  const firstObserved = values[observedIndexes[0]] as number;
  let lastObserved = firstObserved;

  return values.map((value) => {
    if (isObservedMilk(value)) {
      lastObserved = value;
      return value;
    }
    return lastObserved;
  });
}

function backwardFill(values: readonly (number | null | undefined)[]): number[] {
  const observedIndexes = getObservedIndexes(values);
  if (observedIndexes.length === 0) return zeroFill(values);

  const lastObserved = values[observedIndexes[observedIndexes.length - 1]] as number;
  let nextObserved = lastObserved;
  const filled = new Array<number>(values.length);

  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index];
    if (isObservedMilk(value)) {
      nextObserved = value;
      filled[index] = value;
    } else {
      filled[index] = nextObserved;
    }
  }

  return filled;
}

function linearInterpolate(values: readonly (number | null | undefined)[]): number[] {
  const observedIndexes = getObservedIndexes(values);
  if (observedIndexes.length === 0) return zeroFill(values);
  if (observedIndexes.length === 1) {
    return values.map(() => values[observedIndexes[0]] as number);
  }

  const filled = new Array<number>(values.length);

  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (isObservedMilk(value)) {
      filled[index] = value;
      continue;
    }

    const nextObservedIndex = observedIndexes.find((observedIndex) => observedIndex > index);
    const previousObservedIndex = [...observedIndexes]
      .reverse()
      .find((observedIndex) => observedIndex < index);

    if (previousObservedIndex === undefined) {
      filled[index] = values[nextObservedIndex as number] as number;
      continue;
    }
    if (nextObservedIndex === undefined) {
      filled[index] = values[previousObservedIndex] as number;
      continue;
    }

    const previousValue = values[previousObservedIndex] as number;
    const nextValue = values[nextObservedIndex] as number;
    const span = nextObservedIndex - previousObservedIndex;
    const ratio = (index - previousObservedIndex) / span;
    filled[index] = previousValue + (nextValue - previousValue) * ratio;
  }

  return filled;
}

export function prepareDailyModelInput(
  milk: readonly (number | null)[],
  { useImputation, imputationMethod, maxDays = DAILY_MODEL_INPUT_DAYS }: DailyModelInputOptions
): DailyModelInput {
  const values = milk.slice(0, maxDays);
  const missingCount = values.filter((value) => !isObservedMilk(value)).length;

  const preparedMilk = useImputation
    ? imputationMethod === "forward_fill"
      ? forwardFill(values)
      : imputationMethod === "backward_fill"
        ? backwardFill(values)
        : linearInterpolate(values)
    : zeroFill(values);

  return {
    dim: preparedMilk.map((_, index) => index + 1),
    milk: preparedMilk,
    missingCount,
  };
}

export function prepareObservedDailyModelInput(
  milk: readonly (number | null)[],
  maxDays = DAILY_MODEL_INPUT_DAYS
): DailyModelInput {
  const values = milk.slice(0, maxDays);
  const observed = values
    .map((value, index) => (isObservedMilk(value) ? { dim: index + 1, milk: value } : null))
    .filter((value): value is { dim: number; milk: number } => value !== null);

  return {
    dim: observed.map((value) => value.dim),
    milk: observed.map((value) => value.milk),
    missingCount: values.length - observed.length,
  };
}

export function preparePeriodicModelInput(
  dim: readonly number[],
  milkrecordings: readonly number[],
  maxDays = DAILY_MODEL_INPUT_DAYS
): DailyModelInput {
  const projected = Array.from({ length: maxDays }, () => 0);
  dim.forEach((day, index) => {
    if (day >= 1 && day <= maxDays) {
      projected[day - 1] = milkrecordings[index] ?? 0;
    }
  });

  return {
    dim: Array.from({ length: maxDays }, (_, index) => index + 1),
    milk: projected,
    missingCount: maxDays - new Set(dim.filter((day) => day >= 1 && day <= maxDays)).size,
  };
}
