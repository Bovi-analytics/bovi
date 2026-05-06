"use client";

import type { ReactElement } from "react";
import { Anchor, Badge, Card, Divider, Group, Loader, Stack, Text } from "@mantine/core";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ComparisonResults } from "../components/comparison-results";
import { SubmissionForm } from "../components/submission-form";
import { useChallenge } from "../hooks/use-challenges";
import { useSubmissions } from "../hooks/use-submissions";

const DATASET_LABEL: Record<string, string> = {
  icar: "ICAR cohort",
  upload: "Custom upload",
};

export default function ChallengeDetailPage(): ReactElement {
  const { id } = useParams<{ id: string }>();
  const challengeId = Number(id);
  const { data: challenge, isLoading: challengeLoading } = useChallenge(challengeId);
  const { data: submissions, isLoading: submissionsLoading, refetch } = useSubmissions();

  const challengeSubmissions = submissions?.filter((s) => s.challenge_id === challengeId) ?? [];
  const latest = challengeSubmissions[0];

  if (challengeLoading || submissionsLoading) return <Loader />;

  return (
    <div className="space-y-6 p-6">
      <Anchor component={Link} href="/benchmark" size="sm">
        <Group gap={4} wrap="nowrap">
          <ArrowLeft size={14} />
          <span>Back to all challenges</span>
        </Group>
      </Anchor>

      <Stack gap={4}>
        <Group gap="sm" align="center">
          <h1 className="text-2xl font-semibold">Challenge #{challengeId}</h1>
          {challenge && (
            <>
              <Badge color="gray" variant="light">
                {challenge.name ?? DATASET_LABEL[challenge.dataset] ?? challenge.dataset}
              </Badge>
              <Badge color="gray" variant="light">
                {DATASET_LABEL[challenge.dataset] ?? challenge.dataset}
              </Badge>
            </>
          )}
        </Group>
        <Text size="sm" c="dimmed">
          Submit a challenger - a Bovi model or your own CSV - to compare against the fixed TIM
          (ICAR) benchmark for this cohort.
        </Text>
      </Stack>

      <Card withBorder padding="md">
        <Stack gap={4} mb="sm">
          <Text fw={600}>Submit a challenger</Text>
          <Text size="xs" c="dimmed">
            Pick a Bovi model to run, or upload the results of your own calculation. Bovi compares
            it against the TIM (ICAR) benchmark for this challenge.
          </Text>
        </Stack>
        <SubmissionForm
          challengeId={challengeId}
          onSuccess={() => { refetch(); }}
        />
      </Card>

      {latest && (
        <>
          <Divider />
          <Card withBorder padding="md">
            <ComparisonResults submission={latest} />
          </Card>
        </>
      )}

      {challengeSubmissions.length > 1 && (
        <Text size="xs" c="dimmed">{challengeSubmissions.length} submissions total for this challenge.</Text>
      )}
    </div>
  );
}
