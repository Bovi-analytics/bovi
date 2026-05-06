"use client";

import { useMemo } from "react";
import type { ReactElement } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { useWeightUnit } from "@/app/providers/unit-provider";
import { convertWeight } from "@/lib/units";
import type { WeightUnit } from "@/lib/units";

/* ------------------------------------------------------------------ */
/*  Type definitions for the chart's input data                        */
/* ------------------------------------------------------------------ */

export interface CurvePoint {
  readonly dim: number;
  readonly yield: number;
}

export interface CurveData {
  readonly name: string;
  readonly color: string;
  readonly data: readonly CurvePoint[];
}

export interface Observation {
  readonly dim: number;
  readonly yield: number;
}

export interface Annotation {
  readonly dim: number;
  readonly yield: number;
  readonly label: string;
}

interface LactationCurveChartProps {
  readonly curves?: readonly CurveData[];
  readonly observations?: readonly Observation[];
  readonly annotations?: readonly Annotation[];
  readonly height?: number;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function convertPoints<T extends { readonly yield: number }>(
  points: readonly T[],
  unit: WeightUnit
): T[] {
  if (unit === "kg") return points as T[];
  return points.map((p) => ({ ...p, yield: convertWeight(p.yield, unit) }));
}

function formatTooltipValue(value: number, unit: WeightUnit): string {
  if (unit === "lbs") {
    return value.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  }
  return value.toFixed(1);
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function LactationCurveChart({
  curves = [],
  observations = [],
  annotations = [],
  height = 400,
}: LactationCurveChartProps): ReactElement {
  const { weightUnit } = useWeightUnit();
  const yAxisLabel = weightUnit === "lbs" ? "Milk Yield (lbs/day)" : "Milk Yield (kg/day)";

  const convertedCurves = useMemo(
    () => curves.map((c) => ({ ...c, data: convertPoints(c.data, weightUnit) })),
    [curves, weightUnit]
  );
  const convertedObs = useMemo(() => convertPoints(observations, weightUnit), [observations, weightUnit]);
  const convertedAnnotations = useMemo(() => convertPoints(annotations, weightUnit), [annotations, weightUnit]);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />

        <XAxis
          dataKey="dim"
          type="number"
          domain={[0, "dataMax"]}
          label={{ value: "Days in Milk", position: "bottom", offset: 0 }}
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
        />

        <YAxis
          dataKey="yield"
          type="number"
          label={{
            value: yAxisLabel,
            angle: -90,
            position: "insideLeft",
            offset: 10,
          }}
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
        />

        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "0.5rem",
            color: "hsl(var(--card-foreground))",
            fontSize: 12,
          }}
          formatter={(value: number) => [
            formatTooltipValue(value, weightUnit),
            yAxisLabel,
          ]}
        />

        <Legend verticalAlign="top" height={36} />

        {/* Fitted model curves - one Line per model */}
        {convertedCurves.map((curve) => (
          <Line
            key={curve.name}
            data={curve.data as CurvePoint[]}
            dataKey="yield"
            name={curve.name}
            stroke={curve.color}
            strokeWidth={2}
            dot={false}
            type="monotone"
          />
        ))}

        {/* Raw observations - scatter points */}
        {convertedObs.length > 0 && (
          <Scatter
            data={[...convertedObs]}
            dataKey="yield"
            name="Observations"
            fill="hsl(var(--foreground))"
            opacity={0.7}
          />
        )}

        {/* Annotations - highlighted scatter points with labels */}
        {convertedAnnotations.length > 0 && (
          <Scatter
            data={[...convertedAnnotations]}
            dataKey="yield"
            name="Annotations"
            fill="hsl(var(--accent))"
            shape="diamond"
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
