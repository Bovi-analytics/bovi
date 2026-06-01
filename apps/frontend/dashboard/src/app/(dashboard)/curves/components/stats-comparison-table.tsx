"use client";

import type { ReactElement } from "react";
import { Table, Tooltip } from "@mantine/core";
import { Info } from "lucide-react";
import { useWeightUnit } from "@/app/providers/unit-provider";
import { formatCharacteristicValue, isWeightCharacteristic } from "@/lib/characteristics";
import { getUnitLabel } from "@/lib/units";
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
  name,
  value,
  isLoading,
  weightUnit,
  unit,
}: {
  name: string;
  value: number | null;
  isLoading: boolean;
  weightUnit: WeightUnit;
  unit: string;
}): ReactElement {
  if (isLoading) {
    return <span className="text-muted-foreground/50">...</span>;
  }
  if (value === null) {
    return <span className="text-muted-foreground">-</span>;
  }

  const isWeight = isWeightCharacteristic(name);
  const displayValue = formatCharacteristicValue(name, value, weightUnit);
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
            <Table.Th>
              <span className="inline-flex items-center gap-1">
                Persistency
                <Tooltip
                  label="Average slope from time to peak until 305 days in milk"
                  withArrow
                  multiline
                  w={230}
                >
                  <Info size={12} className="cursor-help text-muted-foreground" />
                </Tooltip>
              </span>
            </Table.Th>
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
                  name="peak_yield"
                  value={row.peakYield}
                  isLoading={row.isLoading}
                  weightUnit={weightUnit}
                  unit="kg/day"
                />
              </Table.Td>
              <Table.Td>
                <CellValue
                  name="time_to_peak"
                  value={row.timeToPeak}
                  isLoading={row.isLoading}
                  weightUnit={weightUnit}
                  unit="days"
                />
              </Table.Td>
              <Table.Td>
                <CellValue
                  name="cumulative_milk_yield"
                  value={row.cumulativeYield}
                  isLoading={row.isLoading}
                  weightUnit={weightUnit}
                  unit="kg"
                />
              </Table.Td>
              <Table.Td>
                <CellValue
                  name="persistency"
                  value={row.persistency}
                  isLoading={row.isLoading}
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
