"use client";

import type { ReactElement } from "react";
import { Button, Grid, Group, Loader, Stack, Text } from "@mantine/core";
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
    <div className="space-y-6 p-6">
      <Group justify="space-between" align="center">
        <Stack gap={2}>
          <h1 className="text-2xl font-semibold">Benchmark</h1>
          <Text size="sm" c="dimmed">
            Create a challenge from a preset dataset, submit your 305-day yield calculations, and
            compare against ICAR reference values.
          </Text>
        </Stack>
        <Button leftSection={<Plus size={14} />} onClick={() => router.push("/benchmark/new")}>
          New Challenge
        </Button>
      </Group>

      {challenges && challenges.length === 0 && (
        <Text c="dimmed" size="sm">
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
