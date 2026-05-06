"use client";

import type { ReactElement } from "react";
import { Badge, Button, Card, Group, Select, Stack, Text, TextInput } from "@mantine/core";
import { useState } from "react";
import { useSubmitBoviModel } from "../hooks/use-submissions";
import type { BenchmarkModel } from "@/types/api";

interface Props {
  challengeId: number;
  onSuccess: () => void;
}

const MODEL_OPTIONS: { value: BenchmarkModel; label: string }[] = [
  { value: "tim", label: "TIM (ICAR Test Interval Method)" },
  { value: "wood", label: "Wood" },
  { value: "wilmink", label: "Wilmink" },
  { value: "ali_schaeffer", label: "Ali-Schaeffer" },
  { value: "fischer", label: "Fischer" },
  { value: "milkbot", label: "MilkBot" },
  { value: "autoencoder", label: "Autoencoder (deep learning)" },
];

export function SubmissionFormBovi({ challengeId, onSuccess }: Props): ReactElement {
  const [challenger, setChallenger] = useState<BenchmarkModel>("wood");
  const [benchmark, setBenchmark] = useState<BenchmarkModel>("tim");
  const [organization, setOrganization] = useState("");
  const [country, setCountry] = useState("");
  const { mutate, isPending, error } = useSubmitBoviModel(challengeId);

  const same = challenger === benchmark;

  function handleSubmit() {
    if (same) return;
    mutate({ challenger, benchmark, organization, country }, { onSuccess });
  }

  return (
    <Stack gap="md">
      <Group grow align="stretch">
        <Card withBorder padding="sm" bg="var(--mantine-color-violet-light)">
          <Stack gap={6}>
            <Group gap="xs">
              <Text size="xs" tt="uppercase" fw={700} c="violet">Challenger</Text>
              <Badge color="violet" variant="filled">Bovi model</Badge>
            </Group>
            <Select
              label="Pick a model"
              data={MODEL_OPTIONS}
              value={challenger}
              onChange={(v) => v && setChallenger(v as BenchmarkModel)}
            />
          </Stack>
        </Card>

        <Card withBorder padding="sm" bg="var(--mantine-color-blue-light)">
          <Stack gap={6}>
            <Group gap="xs">
              <Text size="xs" tt="uppercase" fw={700} c="blue">Benchmark</Text>
              <Badge color="blue" variant="filled">Bovi model</Badge>
            </Group>
            <Select
              label="Pick a model"
              data={MODEL_OPTIONS}
              value={benchmark}
              onChange={(v) => v && setBenchmark(v as BenchmarkModel)}
            />
          </Stack>
        </Card>
      </Group>

      {same && (
        <Text c="red" size="xs">Benchmark and challenger must differ.</Text>
      )}

      <Group grow>
        <TextInput
          label="Organization (optional)"
          value={organization}
          onChange={(e) => setOrganization(e.target.value)}
        />
        <TextInput
          label="Country (optional)"
          value={country}
          onChange={(e) => setCountry(e.target.value)}
        />
      </Group>

      {error && <Text c="red" size="xs">{(error as Error).message}</Text>}

      <Button onClick={handleSubmit} loading={isPending} disabled={same}>
        Run &amp; Compare
      </Button>
    </Stack>
  );
}
