"use client";

import type { ReactElement } from "react";
import { Card, Divider, Group, Loader, Stack, Text } from "@mantine/core";
import { useParams } from "next/navigation";
import { ComparisonResults } from "../components/comparison-results";
import { SubmissionForm } from "../components/submission-form";
import { useSubmissions } from "../hooks/use-submissions";

export default function ChallengeDetailPage(): ReactElement {
  const { id } = useParams<{ id: string }>();
  const challengeId = Number(id);
  const { data: submissions, isLoading, refetch } = useSubmissions();

  const challengeSubmissions = submissions?.filter((s) => s.challenge_id === challengeId) ?? [];
  const latest = challengeSubmissions[0];

  if (isLoading) return <Loader />;

  return (
    <div className="space-y-6 p-6">
      <Stack gap={2}>
        <Group>
          <h1 className="text-2xl font-semibold">Challenge #{challengeId}</h1>
        </Group>
        <Text size="sm" c="dimmed">
          Submit your 305-day yield calculations using a Bovi model, or upload results from your
          own method as a CSV. Then download the comparison report.
        </Text>
      </Stack>

      <Card withBorder padding="md">
        <Text fw={600} mb="sm">Submit Results</Text>
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
