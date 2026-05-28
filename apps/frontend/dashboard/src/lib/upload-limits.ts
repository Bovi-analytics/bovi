export const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;
export const MAX_UPLOAD_LABEL = "500 MB";
export const UPLOAD_LIMIT_DESCRIPTION = `Maximum file size: ${MAX_UPLOAD_LABEL}. For larger exports, split the CSV into smaller files.`;

export function uploadSizeErrorMessage(file: Pick<File, "name" | "size">): string {
  return `'${file.name || "CSV file"}' is larger than the ${MAX_UPLOAD_LABEL} upload limit. Split the file into smaller CSV files and upload them separately.`;
}

export function assertUploadSize(file: Pick<File, "name" | "size">): void {
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new Error(uploadSizeErrorMessage(file));
  }
}
