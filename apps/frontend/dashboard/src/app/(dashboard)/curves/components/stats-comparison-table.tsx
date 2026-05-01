"use client";

import type { ReactElement } from "react";
import { Table } from "@mantine/core";
import { useWeightUnit } from "@/app/providers/unit-provider";
import { formatWeight, getUnitLabel } from "@/lib/units";
import type { WeightUnit } from "@/lib/units";

export interface StatsRow {
  readonly name: string;
  readonly color: string;
  readonly peakYield: number | null;
  readonly timeToPeak: number | null;
  readonly cumulativeYield: number | null;
  readonly persistency: number | null;
  readonly isLoading: boolean;
}

interface StatsComparisonTableProps {
  readonly rows: StatsRow[];
}

function CellValue({
  value,
  isLoading,
  isWeight,
  weightUnit,
  unit,
}: {
  value: number | null;
  isLoading: boolean;
  isWeight: boolean;
  weightUnit: WeightUnit;
  unit: string;
}): ReactElement {
  if (isLoading) {
    return <span className="text-muted-foreground/50">...</span>;
  }
  if (value === null) {
    return <span className="text-muted-foreground">—</span>;
  }

  const displayValue = isWeight ? formatWeight(value, weightUnit) : value.toFixed(1);
  const displayUnit = isWeight ? getUnitLabel(unit, weightUnit) : unit;

  return (
    <span>
      {displayValue}
      {displayUnit && <span className="ml-1 text-xs text-muted-foreground">{displayUnit}</span>}
    </span>
  );
}

export function StatsComparisonTable({ rows }: StatsComparisonTableProps): ReactElement {
  const { weightUnit } = useWeightUnit();

  if (rows.length === 0) {
    return <></>;
  }

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-3 text-sm font-medium text-muted-foreground">Characteristics</h3>
      <Table horizontalSpacing="sm" verticalSpacing="xs" withTableBorder={false}>
        <Table.Thead>
          <Table.Tr className="text-xs text-muted-foreground">
            <Table.Th>Model</Table.Th>
            <Table.Th>Peak Yield</Table.Th>
            <Table.Th>Time to Peak</Table.Th>
            <Table.Th>Cumul. Yield</Table.Th>
            <Table.Th>Persistency</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map((row) => (
            <Table.Tr key={row.name} className="text-sm">
              <Table.Td>
                <span className="inline-flex items-center gap-2">
                  <span
                    className="inline-block h-3 w-3 rounded-full"
                    style={{ backgroundColor: row.color }}
                  />
                  {row.name}
                </span>
              </Table.Td>
              <Table.Td>
                <CellValue
                  value={row.peakYield}
                  isLoading={row.isLoading}
                  isWeight
                  weightUnit={weightUnit}
                  unit="kg/day"
                />
              </Table.Td>
              <Table.Td>
                <CellValue
                  value={row.timeToPeak}
                  isLoading={row.isLoading}
                  isWeight={false}
                  weightUnit={weightUnit}
                  unit="days"
                />
              </Table.Td>
              <Table.Td>
                <CellValue
                  value={row.cumulativeYield}
                  isLoading={row.isLoading}
                  isWeight
                  weightUnit={weightUnit}
                  unit="kg"
                />
              </Table.Td>
              <Table.Td>
                <CellValue
                  value={row.persistency}
                  isLoading={row.isLoading}
                  isWeight={false}
                  weightUnit={weightUnit}
                  unit=""
                />
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </div>
  );
}
