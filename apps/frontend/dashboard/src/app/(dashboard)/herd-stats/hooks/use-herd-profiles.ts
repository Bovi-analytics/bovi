"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createHerdProfile,
  deleteHerdProfile,
  listHerdProfiles,
  updateHerdProfile,
} from "@/lib/api-client";
import { useAuth } from "@/lib/auth";
import type { HerdProfileCreate } from "@/types/api";

const QUERY_KEY = ["herd-profiles"] as const;

export function useHerdProfiles() {
  const { selectedOrganizationId } = useAuth();
  return useQuery({
    queryKey: [...QUERY_KEY, selectedOrganizationId],
    queryFn: () => listHerdProfiles(selectedOrganizationId ?? 0),
    enabled: selectedOrganizationId !== null,
  });
}

export function useCreateHerdProfile() {
  const qc = useQueryClient();
  const { selectedOrganizationId } = useAuth();
  return useMutation({
    mutationFn: (data: Omit<HerdProfileCreate, "organization_id">) => {
      if (typeof selectedOrganizationId !== "number") {
        throw new Error("Select a specific organization before creating a herd profile.");
      }
      return createHerdProfile({ ...data, organization_id: selectedOrganizationId });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  });
}

export function useUpdateHerdProfile() {
  const qc = useQueryClient();
  const { selectedOrganizationId } = useAuth();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: Omit<HerdProfileCreate, "organization_id"> }) => {
      if (typeof selectedOrganizationId !== "number") {
        throw new Error("Select a specific organization before updating a herd profile.");
      }
      return updateHerdProfile(id, { ...data, organization_id: selectedOrganizationId });
    },
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
