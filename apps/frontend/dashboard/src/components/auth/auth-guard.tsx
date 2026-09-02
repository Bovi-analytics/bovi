"use client";

import type { ReactNode } from "react";
import {
  Alert,
  Button,
  Checkbox,
  Modal,
  ScrollArea,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { CenteredLoader } from "@/components/dashboard/centered-loader";
import { CURRENT_TERMS, TERMS_OF_USE_PARAGRAPHS } from "@/data/terms-of-use";
import { acceptCurrentTerms, createOrganization } from "@/lib/api-client";
import { useAuth } from "@/lib/auth";

export function AuthGuard({ children }: { readonly children: ReactNode }): ReactNode {
  const { isAuthenticated, isLoading, setSelectedOrganizationId, user } = useAuth();
  const router = useRouter();
  const [organizationName, setOrganizationName] = useState("");
  const [isCreatingOrganization, setIsCreatingOrganization] = useState(false);
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const [isAcceptingTerms, setIsAcceptingTerms] = useState(false);
  const [termsError, setTermsError] = useState<string | null>(null);
  const suggestedName = "Your Organization";

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
  if (user && !user.terms_acceptance.accepted) {
    return (
      <Modal
        opened
        onClose={() => undefined}
        closeOnClickOutside={false}
        closeOnEscape={false}
        withCloseButton={false}
        title={CURRENT_TERMS.title}
        size="xl"
      >
        <Stack gap="md">
          <Text size="sm" c="dimmed">
            You must accept the current Terms of Use and Data Contribution Agreement before using
            the Bovi dashboard.
          </Text>
          <ScrollArea h={360} offsetScrollbars type="always">
            <Stack gap="sm" pr="md">
              {TERMS_OF_USE_PARAGRAPHS.map((paragraph) => (
                <Text key={paragraph} size="sm" ta="justify" lh={1.6}>
                  {paragraph}
                </Text>
              ))}
            </Stack>
          </ScrollArea>
          <Text size="xs" c="dimmed">
            Document version {CURRENT_TERMS.version}.
          </Text>
          <Checkbox
            checked={acceptedTerms}
            onChange={(event) => setAcceptedTerms(event.currentTarget.checked)}
            label="I have read and agree to the Terms of Use and Data Contribution Agreement."
          />
          {termsError && (
            <Alert color="red" variant="light">
              {termsError}
            </Alert>
          )}
          <Button
            disabled={!acceptedTerms}
            loading={isAcceptingTerms}
            onClick={async () => {
              setIsAcceptingTerms(true);
              setTermsError(null);
              try {
                await acceptCurrentTerms();
                window.location.reload();
              } catch (error) {
                setTermsError(error instanceof Error ? error.message : String(error));
              } finally {
                setIsAcceptingTerms(false);
              }
            }}
          >
            Accept and continue
          </Button>
        </Stack>
      </Modal>
    );
  }
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
