"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createChallenge, listChallenges } from "@/lib/api-client";
import type { ChallengeCreate } from "@/types/api";

const KEY = ["benchmark-challenges"] as const;

export function useChallenges() {
  return useQuery({ queryKey: KEY, queryFn: listChallenges });
}

export function useCreateChallenge() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: ChallengeCreate) => createChallenge(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}
