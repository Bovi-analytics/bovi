import type { ReactElement } from "react";

const CHARACTERISTIC_META: Record<string, { label: string; unit: string }> = {
  peak_yield: { label: "Peak Yield", unit: "kg/day" },
  time_to_peak: { label: "Time to Peak", unit: "days" },
  cumulative_milk_yield: { label: "Cumulative Yield", unit: "kg" },
  persistency: { label: "Persistency", unit: "" },
};

interface StatCardProps {
  readonly name: string;
  readonly value: number | null;
  readonly isLoading: boolean;
}

export function StatCard({ name, value, isLoading }: StatCardProps): ReactElement {
  const meta = CHARACTERISTIC_META[name] ?? { label: name, unit: "" };

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <p className="text-sm font-medium text-muted-foreground">{meta.label}</p>
      <div className="mt-2">
        {isLoading ? (
          <p className="text-2xl font-bold text-muted-foreground/50">…</p>
        ) : value !== null ? (
          <p className="text-2xl font-bold text-foreground">
            {value.toFixed(1)}
            {meta.unit && (
              <span className="ml-1 text-sm font-normal text-muted-foreground">{meta.unit}</span>
            )}
          </p>
        ) : (
          <p className="text-2xl font-bold text-destructive">—</p>
        )}
      </div>
    </div>
  );
}
