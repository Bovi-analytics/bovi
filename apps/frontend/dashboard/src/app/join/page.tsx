"use client";

import { useEffect, useState } from "react";
import type { ReactElement } from "react";
import { Button, Loader, Stack, Text, Title } from "@mantine/core";
import { useRouter, useSearchParams } from "next/navigation";
import { acceptInvite } from "@/lib/api-client";
import { startLogin, useAuth } from "@/lib/auth";

const PENDING_INVITE_KEY = "bovi:pending-invite";

export default function JoinPage(): ReactElement {
  const params = useSearchParams();
  const router = useRouter();
  const { isAuthenticated, isLoading, setSelectedOrganizationId } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const token = params.get("invite");

  useEffect(() => {
    if (!token) {
      setError("Invite link is missing.");
      return;
    }
    window.localStorage.setItem(PENDING_INVITE_KEY, token);
    if (!isLoading && !isAuthenticated) {
      void startLogin();
    }
  }, [isAuthenticated, isLoading, token]);

  useEffect(() => {
    if (!isAuthenticated || isLoading) return;
    const pendingToken = token ?? window.localStorage.getItem(PENDING_INVITE_KEY);
    if (!pendingToken) return;
    const run = async () => {
      try {
        const organization = await acceptInvite(pendingToken);
        window.localStorage.removeItem(PENDING_INVITE_KEY);
        setSelectedOrganizationId(organization.id);
        router.push("/");
      } catch {
        setError("Invite link is expired, revoked, or invalid.");
      }
    };
    void run();
  }, [isAuthenticated, isLoading, router, setSelectedOrganizationId, token]);

  if (!error) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <Stack maw={420} gap="md">
        <Title order={1}>Unable to join organization</Title>
        <Text>{error}</Text>
        <Button onClick={() => router.push("/")}>Continue</Button>
      </Stack>
    </div>
  );
}
