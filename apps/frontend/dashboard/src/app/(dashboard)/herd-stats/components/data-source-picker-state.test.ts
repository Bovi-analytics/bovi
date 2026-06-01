import { describe, expect, test } from "bun:test";
import type { ActivePreset, UploadedDataset } from "@/app/providers/uploaded-cows-provider";
import { getInitialDataSource } from "./data-source-picker-state";

const UPLOADED_DATASET: UploadedDataset = {
  id: "upload-1",
  name: "Uploaded herd",
  format: "icar_test_day",
  uploadedAt: "2026-05-29T00:00:00.000Z",
  cows: [],
};

const ACTIVE_PRESET: ActivePreset = {
  dataset: "sunnyside",
  size: "small",
  period: "mixed",
};

describe("getInitialDataSource", () => {
  test("opens preset datasets by default without selecting a dataset", () => {
    expect(getInitialDataSource(null, null)).toBe("aurora");
  });

  test("keeps the current in-memory upload section open after a deliberate upload", () => {
    expect(getInitialDataSource(null, UPLOADED_DATASET)).toBe("upload");
  });

  test("keeps the current in-memory preset section open after a deliberate preset selection", () => {
    expect(getInitialDataSource(ACTIVE_PRESET, UPLOADED_DATASET)).toBe("sunnyside");
  });
});
