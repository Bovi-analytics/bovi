"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createHerdProfile,
  deleteHerdProfile,
  listHerdProfiles,
  updateHerdProfile,
} from "@/lib/api-client";
import type { HerdProfileCreate } from "@/types/api";

const QUERY_KEY = ["herd-profiles"] as const;

export function useHerdProfiles() {
  return useQuery({
    queryKey: QUERY_KEY,
    queryFn: listHerdProfiles,
  });
}

export function useCreateHerdProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: HerdProfileCreate) => createHerdProfile(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  });
}

export function useUpdateHerdProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: HerdProfileCreate }) =>
      updateHerdProfile(id, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  });
}

export function useDeleteHerdProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => deleteHerdProfile(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  });
}
