import { describe, expect, test } from "bun:test";
import { assertUploadSize, MAX_UPLOAD_BYTES, uploadSizeErrorMessage } from "./upload-limits";

describe("upload limits", () => {
  test("allows files at the configured limit", () => {
    expect(() => assertUploadSize({ name: "allowed.csv", size: MAX_UPLOAD_BYTES })).not.toThrow();
  });

  test("rejects files above the configured limit with a split-file message", () => {
    const file = { name: "large.csv", size: MAX_UPLOAD_BYTES + 1 };

    expect(() => assertUploadSize(file)).toThrow("500 MB upload limit");
    expect(uploadSizeErrorMessage(file)).toContain("Split the file into smaller CSV files");
  });
});
