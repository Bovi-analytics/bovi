"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createChallengePreset,
  createChallengeUpload,
  getChallenge,
  listChallenges,
} from "@/lib/api-client";
import { useAuth } from "@/lib/auth";
import type { ChallengeCreatePreset } from "@/types/api";

const KEY = ["benchmark-challenges"] as const;

export function useChallenges() {
  const { selectedOrganizationId } = useAuth();
  return useQuery({
    queryKey: [...KEY, selectedOrganizationId],
    queryFn: () => listChallenges(selectedOrganizationId ?? 0),
    enabled: selectedOrganizationId !== null,
  });
}

export function useChallenge(id: number) {
  return useQuery({
    queryKey: [...KEY, id] as const,
    queryFn: () => getChallenge(id),
    enabled: Number.isFinite(id),
  });
}

export function useCreateChallengePreset() {
  const qc = useQueryClient();
  const { selectedOrganizationId } = useAuth();
  return useMutation({
    mutationFn: (data: Omit<ChallengeCreatePreset, "organization_id">) => {
      if (typeof selectedOrganizationId !== "number") {
        throw new Error("Select a specific organization before creating a challenge.");
      }
      return createChallengePreset({ ...data, organization_id: selectedOrganizationId });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}

export function useCreateChallengeUpload() {
  const qc = useQueryClient();
  const { selectedOrganizationId } = useAuth();
  return useMutation({
    mutationFn: ({
      name,
      testDayCsv,
      actualYieldsCsv,
    }: {
      name: string;
      testDayCsv: File;
      actualYieldsCsv: File;
    }) => {
      if (typeof selectedOrganizationId !== "number") {
        throw new Error("Select a specific organization before uploading a challenge.");
      }
      return createChallengeUpload(name, testDayCsv, actualYieldsCsv, selectedOrganizationId);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}
