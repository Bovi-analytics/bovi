"use client";

import type { ReactNode } from "react";
import { Button, Stack, TextInput, Title } from "@mantine/core";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { CenteredLoader } from "@/components/dashboard/centered-loader";
import { createOrganization } from "@/lib/api-client";
import { useAuth } from "@/lib/auth";

export function AuthGuard({ children }: { readonly children: ReactNode }): ReactNode {
  const { isAuthenticated, isLoading, setSelectedOrganizationId, user } = useAuth();
  const router = useRouter();
  const [organizationName, setOrganizationName] = useState("");
  const [isCreatingOrganization, setIsCreatingOrganization] = useState(false);
  const suggestedName = useMemo(() => {
    if (user?.email?.includes("@")) return user.email.split("@")[1];
    return user?.name ? `${user.name}'s organization` : "My organization";
  }, [user]);

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      const nextPath =
        typeof window === "undefined" ? "" : `${window.location.pathname}${window.location.search}`;
      const loginUrl = nextPath
        ? `/auth/login?next=${encodeURIComponent(nextPath)}`
        : "/auth/login";
      router.push(loginUrl);
    }
  }, [isAuthenticated, isLoading, router]);

  if (isLoading) return <CenteredLoader label="Opening your workspace..." />;
  if (!isAuthenticated) return null;
  if (user && user.organizations.length === 0) {
    return (
      <div className="flex min-h-screen items-center justify-center p-6">
        <Stack w="100%" maw={420} gap="md">
          <Title order={1}>Create your Bovi organization</Title>
          <TextInput
            label="Organization name"
            value={organizationName}
            placeholder={suggestedName}
            onChange={(event) => setOrganizationName(event.currentTarget.value)}
          />
          <Button
            loading={isCreatingOrganization}
            onClick={async () => {
              setIsCreatingOrganization(true);
              try {
                const organization = await createOrganization(
                  organizationName.trim() || suggestedName
                );
                setSelectedOrganizationId(organization.id);
                window.location.reload();
              } finally {
                setIsCreatingOrganization(false);
              }
            }}
          >
            Create organization
          </Button>
        </Stack>
      </div>
    );
  }
  return children;
}
