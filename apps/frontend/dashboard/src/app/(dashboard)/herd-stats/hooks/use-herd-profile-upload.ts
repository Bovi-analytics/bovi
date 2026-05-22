"use client";

import { useMutation } from "@tanstack/react-query";
import { uploadHerdProfileCsv } from "@/lib/api-client";
import { useAuth } from "@/lib/auth";

export function useHerdProfileUpload() {
  const { selectedOrganizationId } = useAuth();
  return useMutation({
    mutationFn: (file: File) => {
      if (typeof selectedOrganizationId !== "number") {
        throw new Error("Select a specific organization before uploading herd data.");
      }
      return uploadHerdProfileCsv(file, selectedOrganizationId);
    },
  });
}
