import type { ReactElement } from "react";
import type { ModelMetadata } from "@/data/model-metadata";

interface ModelInfoProps {
  readonly model: ModelMetadata;
  readonly fittedValues?: Record<string, number>;
}

export function ModelInfo({ model, fittedValues }: ModelInfoProps): ReactElement {
  return (
    <div className="space-y-4">
      {/* Description */}
      <p className="text-sm text-muted-foreground">{model.description}</p>

      {/* Formula */}
      <div className="rounded-lg bg-muted/50 px-4 py-3">
        <p className="font-mono text-sm text-foreground">{model.formula}</p>
      </div>

      {/* Parameter table */}
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-muted-foreground">
            <th className="pb-2 font-medium">Parameter</th>
            <th className="pb-2 font-medium">Description</th>
            {fittedValues && <th className="pb-2 text-right font-medium">Value</th>}
          </tr>
        </thead>
        <tbody>
          {model.parameters.map((param) => (
            <tr key={param.name} className="border-b border-border/50">
              <td className="py-2 font-mono text-foreground">{param.name}</td>
              <td className="py-2 text-muted-foreground">{param.description}</td>
              {fittedValues && (
                <td className="py-2 text-right font-mono text-foreground">
                  {fittedValues[param.name]?.toFixed(4) ?? "—"}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
