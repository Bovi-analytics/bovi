"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactElement } from "react";
import Image from "next/image";
import {
  Alert,
  Badge,
  Button,
  Group,
  Loader,
  Paper,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import { Building2, CalendarClock, LogIn, ShieldCheck, TriangleAlert } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { acceptInvite, getInvitePreview } from "@/lib/api-client";
import type { OrganizationInvitePreview } from "@/lib/api-client";
import { startLogin, useAuth } from "@/lib/auth";

const PENDING_INVITE_KEY = "bovi:pending-invite";
const PENDING_INVITE_ACCEPT_KEY = "bovi:pending-invite-accept";

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export default function JoinPage(): ReactElement {
  const params = useSearchParams();
  const router = useRouter();
  const { isAuthenticated, isLoading, setSelectedOrganizationId } = useAuth();
  const token = params.get("invite");
  const [preview, setPreview] = useState<OrganizationInvitePreview | null>(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(true);
  const [isAccepting, setIsAccepting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const autoAcceptAttemptedRef = useRef(false);

  const nextPath = useMemo(() => {
    if (!token) return "/join";
    return `/join?invite=${encodeURIComponent(token)}`;
  }, [token]);

  useEffect(() => {
    if (!token) {
      setIsPreviewLoading(false);
      setError("This invite link is missing its token.");
      return;
    }

    window.localStorage.setItem(PENDING_INVITE_KEY, token);
    setIsPreviewLoading(true);
    setError(null);
    void getInvitePreview(token)
      .then((data) => {
        setPreview(data);
      })
      .catch(() => {
        setError("This invite link is expired, revoked, or invalid.");
      })
      .finally(() => {
        setIsPreviewLoading(false);
      });
  }, [token]);

  const acceptCurrentInvite = useCallback(
    async (explicitToken?: string): Promise<void> => {
      const pendingToken =
        explicitToken ?? token ?? window.localStorage.getItem(PENDING_INVITE_KEY);
      if (!pendingToken) {
        setError("This invite link is missing its token.");
        return;
      }

      setIsAccepting(true);
      setError(null);
      try {
        const organization = await acceptInvite(pendingToken);
        window.localStorage.removeItem(PENDING_INVITE_KEY);
        window.localStorage.removeItem(PENDING_INVITE_ACCEPT_KEY);
        setSelectedOrganizationId(organization.id);
        window.location.assign("/");
      } catch {
        window.localStorage.removeItem(PENDING_INVITE_ACCEPT_KEY);
        setError("This invite link is expired, revoked, or invalid.");
      } finally {
        setIsAccepting(false);
      }
    },
    [setSelectedOrganizationId, token]
  );

  useEffect(() => {
    if (
      autoAcceptAttemptedRef.current ||
      !isAuthenticated ||
      isLoading ||
      isPreviewLoading ||
      !preview ||
      error
    ) {
      return;
    }

    const pendingToken = token ?? window.localStorage.getItem(PENDING_INVITE_KEY);
    const tokenToAccept = window.localStorage.getItem(PENDING_INVITE_ACCEPT_KEY);
    if (!pendingToken || tokenToAccept !== pendingToken) return;

    autoAcceptAttemptedRef.current = true;
    void acceptCurrentInvite(pendingToken);
  }, [acceptCurrentInvite, error, isAuthenticated, isLoading, isPreviewLoading, preview, token]);

  async function signInToAccept(): Promise<void> {
    if (token) {
      window.localStorage.setItem(PENDING_INVITE_KEY, token);
      window.localStorage.setItem(PENDING_INVITE_ACCEPT_KEY, token);
    }
    await startLogin(nextPath);
  }

  return (
    <main className="min-h-screen bg-background px-6 py-8 text-foreground">
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-5xl flex-col">
        <Group justify="space-between" align="center">
          <Image
            src="/bovi-logo.png"
            alt="Bovi-Analytics"
            width={2255}
            height={699}
            priority
            className="h-auto w-[150px]"
          />
          <Badge variant="light" color="blue">
            Organization invite
          </Badge>
        </Group>

        <div className="flex flex-1 items-center justify-center py-10">
          <Paper withBorder radius="md" p="xl" className="w-full max-w-2xl bg-card/90">
            <Stack gap="lg">
              {isPreviewLoading ? (
                <Stack align="center" gap="md" py="xl">
                  <Loader />
                  <Text c="dimmed">Loading invite...</Text>
                </Stack>
              ) : error ? (
                <>
                  <ThemeIcon size={52} radius="xl" color="red" variant="light">
                    <TriangleAlert size={28} />
                  </ThemeIcon>
                  <Stack gap="xs">
                    <Title order={1}>Unable to open invite</Title>
                    <Text c="dimmed">{error}</Text>
                  </Stack>
                  <Group>
                    <Button variant="light" onClick={() => router.push("/")}>
                      Go to dashboard
                    </Button>
                  </Group>
                </>
              ) : preview ? (
                <>
                  <ThemeIcon size={52} radius="xl" color="blue" variant="light">
                    <Building2 size={28} />
                  </ThemeIcon>
                  <Stack gap="xs">
                    <Title order={1}>Join {preview.organization_name}</Title>
                    <Text c="dimmed">
                      This invite gives you {preview.role} access to the organization's Bovi
                      Analytics workspace.
                    </Text>
                  </Stack>

                  <Stack gap="sm">
                    <Group gap="sm" align="center">
                      <ShieldCheck size={18} className="text-primary" />
                      <Text size="sm">
                        Organization: <strong>{preview.organization_name}</strong>
                      </Text>
                    </Group>
                    <Group gap="sm" align="center">
                      <Building2 size={18} className="text-primary" />
                      <Text size="sm">
                        Access level: <strong>{preview.role}</strong>
                      </Text>
                    </Group>
                    <Group gap="sm" align="center">
                      <CalendarClock size={18} className="text-primary" />
                      <Text size="sm">
                        Invite expires: <strong>{formatDate(preview.expires_at)}</strong>
                      </Text>
                    </Group>
                  </Stack>

                  {!isAuthenticated && (
                    <Alert color="blue" variant="light" icon={<ShieldCheck size={18} />}>
                      Sign in with Microsoft to accept this invite and create your Bovi workspace
                      access.
                    </Alert>
                  )}

                  <Group>
                    {isAuthenticated ? (
                      <Button
                        loading={isLoading || isAccepting}
                        onClick={() => void acceptCurrentInvite()}
                      >
                        Accept invite
                      </Button>
                    ) : (
                      <Button
                        leftSection={<LogIn size={16} />}
                        loading={isLoading}
                        onClick={() => void signInToAccept()}
                      >
                        Sign in to accept invite
                      </Button>
                    )}
                    <Button variant="subtle" onClick={() => router.push("/")}>
                      Not now
                    </Button>
                  </Group>
                </>
              ) : null}
            </Stack>
          </Paper>
        </div>
      </div>
    </main>
  );
}
