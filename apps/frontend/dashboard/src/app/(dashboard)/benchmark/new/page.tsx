"use client";

import type { ReactElement } from "react";
import { Button, Group, Loader, Select, SegmentedControl, Stack, Text } from "@mantine/core";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useCreateChallenge } from "../hooks/use-challenges";

export default function NewChallengePage(): ReactElement {
  const router = useRouter();
  const [dataset, setDataset] = useState<"aurora" | "sunnyside">("aurora");
  const [size, setSize] = useState<"small" | "medium">("small");
  const [period, setPeriod] = useState<"recent" | "old" | "mixed">("recent");

  const { mutate, isPending, error } = useCreateChallenge();

  function handleCreate() {
    mutate(
      { dataset, size, period },
      {
        onSuccess: (c) => router.push(`/benchmark/${c.id}`),
      }
    );
  }

  return (
    <div className="max-w-lg space-y-6 p-6">
      <Stack gap={2}>
        <h1 className="text-2xl font-semibold">New Challenge</h1>
        <Text size="sm" c="dimmed">
          Choose a dataset, size, and period. The system will sample cows and compute reference
          305-day yields via TIM.
        </Text>
      </Stack>

      <Stack gap="md">
        <Select
          label="Dataset"
          data={[
            { value: "aurora", label: "Aurora Ridge" },
            { value: "sunnyside", label: "Sunnyside" },
          ]}
          value={dataset}
          onChange={(v) => v && setDataset(v as typeof dataset)}
        />

        <div>
          <Text size="sm" fw={500} mb={4}>Size</Text>
          <SegmentedControl
            value={size}
            onChange={(v) => setSize(v as typeof size)}
            data={[
              { value: "small", label: "Small (200 cows)" },
              { value: "medium", label: "Medium (1000 cows)" },
            ]}
          />
        </div>

        <div>
          <Text size="sm" fw={500} mb={4}>Period</Text>
          <SegmentedControl
            value={period}
            onChange={(v) => setPeriod(v as typeof period)}
            data={[
              { value: "recent", label: "Recent" },
              { value: "old", label: "Old" },
              { value: "mixed", label: "Mixed" },
            ]}
          />
        </div>

        {error && <Text c="red" size="sm">{(error as Error).message}</Text>}

        <Group>
          <Button onClick={handleCreate} loading={isPending}>
            Create Challenge
          </Button>
          <Button variant="subtle" onClick={() => router.back()}>
            Cancel
          </Button>
        </Group>

        {isPending && (
          <Group gap="xs">
            <Loader size="xs" />
            <Text size="xs" c="dimmed">Computing reference yields — this may take 30–60 seconds…</Text>
          </Group>
        )}
      </Stack>
    </div>
  );
}
