"use client";

import { useMutation } from "@tanstack/react-query";
import { uploadHerdProfileCsv } from "@/lib/api-client";

export function useHerdProfileUpload() {
  return useMutation({
    mutationFn: (file: File) => uploadHerdProfileCsv(file),
  });
}
