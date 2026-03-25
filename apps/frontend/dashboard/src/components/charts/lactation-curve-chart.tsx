"use client";

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
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function LactationCurveChart({
  curves = [],
  observations = [],
  annotations = [],
  height = 400,
}: LactationCurveChartProps): ReactElement {
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
            value: "Milk Yield (kg/day)",
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
        />

        <Legend verticalAlign="top" height={36} />

        {/* Fitted model curves — one Line per model */}
        {curves.map((curve) => (
          <Line
            key={curve.name}
            data={curve.data}
            dataKey="yield"
            name={curve.name}
            stroke={curve.color}
            strokeWidth={2}
            dot={false}
            type="monotone"
          />
        ))}

        {/* Raw observations — scatter points */}
        {observations.length > 0 && (
          <Scatter
            data={[...observations]}
            dataKey="yield"
            name="Observations"
            fill="hsl(var(--foreground))"
            opacity={0.7}
          />
        )}

        {/* Annotations — highlighted scatter points with labels */}
        {annotations.length > 0 && (
          <Scatter
            data={[...annotations]}
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
