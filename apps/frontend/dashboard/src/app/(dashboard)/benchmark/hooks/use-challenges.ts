"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createChallengePreset,
  createChallengeUpload,
  getChallenge,
  listChallenges,
} from "@/lib/api-client";
import type { ChallengeCreatePreset } from "@/types/api";

const KEY = ["benchmark-challenges"] as const;

export function useChallenges() {
  return useQuery({ queryKey: KEY, queryFn: listChallenges });
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
  return useMutation({
    mutationFn: (data: ChallengeCreatePreset) => createChallengePreset(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}

export function useCreateChallengeUpload() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      name,
      testDayCsv,
      actualYieldsCsv,
    }: {
      name: string;
      testDayCsv: File;
      actualYieldsCsv: File;
    }) => createChallengeUpload(name, testDayCsv, actualYieldsCsv),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}
