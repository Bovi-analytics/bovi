"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createChallengeFromSavedDataset,
  createChallengePreset,
  createChallengeUpload,
  getChallenge,
  listOptionsKey,
  listChallenges,
  type OrganizationListOptions,
} from "@/lib/api-client";
import { useAuth } from "@/lib/auth";
import type { ChallengeCreatePreset, ChallengeDatasetSource, ChallengeDetail } from "@/types/api";

const KEY = ["benchmark-challenges"] as const;

export function useChallenges(options: OrganizationListOptions = {}) {
  const { selectedOrganizationId } = useAuth();
  return useQuery({
    queryKey: [...KEY, selectedOrganizationId, listOptionsKey(options)],
    queryFn: () => listChallenges(selectedOrganizationId ?? 0, options),
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

export function useCreateChallengeFromSavedDataset() {
  const qc = useQueryClient();
  const { selectedOrganizationId } = useAuth();
  return useMutation({
    mutationFn: ({
      name,
      cowMetadata,
      actualYields,
      datasetSources,
    }: {
      name: string;
      cowMetadata: ChallengeDetail["cow_metadata"];
      actualYields: NonNullable<ChallengeDetail["actual_yields"]>;
      datasetSources?: ChallengeDatasetSource[];
    }) => {
      if (typeof selectedOrganizationId !== "number") {
        throw new Error("Select a specific organization before creating a challenge.");
      }
      return createChallengeFromSavedDataset(
        name,
        cowMetadata,
        actualYields,
        selectedOrganizationId,
        datasetSources
      );
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}
