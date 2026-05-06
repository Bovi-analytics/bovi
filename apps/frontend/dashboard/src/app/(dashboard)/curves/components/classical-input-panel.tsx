"use client";

import { useState } from "react";
import type { ReactElement } from "react";
import { Checkbox, Modal } from "@mantine/core";
import { Info } from "lucide-react";
import { ALL_MODELS, MODEL_METADATA } from "@/data/model-metadata";
import { ModelInfo } from "@/app/(dashboard)/models/components/model-info";
import type { Model } from "@/types/api";
import type { ModelMetadata } from "@/data/model-metadata";

interface ClassicalInputPanelProps {
  readonly selectedModels: Model[];
  readonly onToggleModel: (model: Model) => void;
}

export function ClassicalInputPanel({
  selectedModels,
  onToggleModel,
}: ClassicalInputPanelProps): ReactElement {
  const [modalModel, setModalModel] = useState<ModelMetadata | null>(null);

  return (
    <>
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-base font-semibold text-foreground">Classical models</h3>
        <div className="space-y-2">
          {ALL_MODELS.map((m) => (
            <div key={m.id} className="flex items-center justify-between">
              <Checkbox
                label={m.name}
                checked={selectedModels.includes(m.id)}
                onChange={() => onToggleModel(m.id)}
                color={m.color}
                size="sm"
              />
              <button
                type="button"
                onClick={() => setModalModel(MODEL_METADATA[m.id])}
                className="ml-2 text-muted-foreground/50 transition-colors hover:text-muted-foreground"
                aria-label={`Info about ${m.name}`}
              >
                <Info size={14} />
              </button>
            </div>
          ))}
        </div>
      </div>

      <Modal
        opened={modalModel !== null}
        onClose={() => setModalModel(null)}
        title={modalModel?.name ?? ""}
        size="lg"
      >
        {modalModel && <ModelInfo model={modalModel} />}
      </Modal>
    </>
  );
}
