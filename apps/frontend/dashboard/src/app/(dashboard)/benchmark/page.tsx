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
            (ALY) on a cohort of cows with daily milk meter data.
          </Text>
        </Stack>
        <Button leftSection={<Plus size={14} />} onClick={() => router.push("/benchmark/new")}>
          New Challenge
        </Button>
      </Group>

      <Alert color="blue" variant="light" title="How the benchmark works">
        <Stack gap={4}>
          <Text size="sm">
            A challenge consists of a group of cows for which the ground-truth ALY (Actual Lactation
            Yield based on daily milk meter recordings) is known, while only sampled test-day milk
            recordings are provided as input. You can use the built-in reference dataset or upload
            your own test-day records together with corresponding daily milk meter data as ground
            truth. The goal is to estimate the 305-day cumulative milk yield for each lactation.
          </Text>
          <Text size="sm">
            For each challenge, you select both a challenger and a benchmark. Benchmarks include
            ICAR-approved cumulative yield calculation methods such as the Test Interval Method,
            Best Prediction, and interpolation using standard lactation curves. You can also use
            lactation curve models such as Wood, Wilmink, Ali-Schaeffer, Fischer, MilkBot, or the
            autoencoder, or upload your own calculations as a CSV file and compare them against the
            built-in methods.
          </Text>
          <Text size="sm">
            Bovi applies both methods to the same sparse test-day input data and evaluates their
            performance against the ground-truth ALY using Pearson correlation, RMSE, MAE, and MAPE,
            both overall and stratified by parity.
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
