"use client";

import type { ReactElement } from "react";
import { Alert, Button, Grid, Group, Loader, Stack, Text } from "@mantine/core";
import { Plus } from "lucide-react";
import { useRouter } from "next/navigation";
import { ChallengeCard } from "./components/challenge-card";
import { useChallenges } from "./hooks/use-challenges";

export default function BenchmarkPage(): ReactElement {
  const router = useRouter();
  const { data: challenges, isLoading, error } = useChallenges();

  if (isLoading) return <Loader />;
  if (error) return <Text c="red">Failed to load challenges.</Text>;

  return (
    <div className="benchmark-page space-y-6 p-6">
      <Group justify="space-between" align="center">
        <Stack gap={2}>
          <h1 className="text-2xl font-semibold">Benchmark</h1>
          <Text size="sm" c="var(--benchmark-muted-text)">
            Compare a 305-day milk yield calculation against ground-truth Actual Lactation Yield
            (ALY) on a cohort of cows with daily-meter data.
          </Text>
        </Stack>
        <Button leftSection={<Plus size={14} />} onClick={() => router.push("/benchmark/new")}>
          New Challenge
        </Button>
      </Group>

      <Alert color="blue" variant="light" title="How the benchmark works">
        <Stack gap={4}>
          <Text size="sm">
            A challenge is a cohort of cows for which the <strong>ground-truth ALY</strong> (Actual
            Lactation Yield from daily-meter recordings) is known. Use the built-in reference
            cohort, or upload your own test-day records together with daily-meter ground truth.
          </Text>
          <Text size="sm">
            On a challenge you pick a <em>challenger</em> and a <em>benchmark</em> - any combination
            of TIM, Wood, Wilmink, Ali-Schaeffer, Fischer, MilkBot, or the AI autoencoder. The
            challenger can also be your own calculation uploaded as a CSV. Bovi runs both on the
            same sparse test-day input and compares each against the ground-truth ALY (Pearson,
            RMSE, MAE, MAPE), overall and per parity.
          </Text>
        </Stack>
      </Alert>

      {challenges && challenges.length === 0 && (
        <Text c="var(--benchmark-muted-text)" size="sm">
          No challenges yet. Create one to get started.
        </Text>
      )}

      <Grid>
        {challenges?.map((c) => (
          <Grid.Col key={c.id} span={{ base: 12, sm: 6, md: 4 }}>
            <ChallengeCard challenge={c} />
          </Grid.Col>
        ))}
      </Grid>
    </div>
  );
}
