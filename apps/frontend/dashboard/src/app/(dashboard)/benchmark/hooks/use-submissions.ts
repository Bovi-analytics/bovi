"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getSubmission, listSubmissions, submitBoviModel, submitOwnMethod } from "@/lib/api-client";
import { useAuth } from "@/lib/auth";

const KEY = ["benchmark-submissions"] as const;

export function useSubmissions() {
  const { selectedOrganizationId } = useAuth();
  return useQuery({
    queryKey: [...KEY, selectedOrganizationId],
    queryFn: () => listSubmissions(selectedOrganizationId ?? 0),
    enabled: selectedOrganizationId !== null,
  });
}

export function useSubmission(id: number) {
  return useQuery({
    queryKey: [...KEY, id],
    queryFn: () => getSubmission(id),
    enabled: !!id,
  });
}

export function useSubmitBoviModel(challengeId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Parameters<typeof submitBoviModel>[1]) =>
      submitBoviModel(challengeId, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}

export function useSubmitOwnMethod(challengeId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ file, meta }: { file: File; meta: Parameters<typeof submitOwnMethod>[2] }) =>
      submitOwnMethod(challengeId, file, meta),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}
