import type { ActivePreset, UploadedDataset } from "@/app/providers/uploaded-cows-provider";
import type { PresetDatasetKey } from "@/types/api";

export type DataSourcePickerSourceKey = PresetDatasetKey | "upload" | "saved";

export function getInitialDataSource(
  activePreset: ActivePreset | null,
  uploadedDataset: UploadedDataset | null
): DataSourcePickerSourceKey {
  if (activePreset) return activePreset.dataset;
  if (uploadedDataset) return "upload";
  return "aurora";
}
