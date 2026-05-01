"use client";

import type { ReactElement } from "react";
import { Button, Select, Stack, Text, TextInput } from "@mantine/core";
import { useState } from "react";
import { useSubmitBoviModel } from "../hooks/use-submissions";

interface Props {
  challengeId: number;
  onSuccess: () => void;
}

export function SubmissionFormBovi({ challengeId, onSuccess }: Props): ReactElement {
  const [modelType, setModelType] = useState("tim");
  const [organization, setOrganization] = useState("");
  const [country, setCountry] = useState("");
  const { mutate, isPending, error } = useSubmitBoviModel(challengeId);

  function handleSubmit() {
    mutate(
      { submission_type: "bovi_model", model_type: modelType, organization, country },
      { onSuccess }
    );
  }

  return (
    <Stack gap="sm">
      <Text size="sm" c="dimmed">
        Bovi will compute 305-day yields for all challenge cows using the selected model.
      </Text>
      <Select
        label="Model"
        data={[{ value: "tim", label: "TIM (ICAR Test Interval Method)" }]}
        value={modelType}
        onChange={(v) => v && setModelType(v)}
      />
      <TextInput label="Organization" value={organization} onChange={(e) => setOrganization(e.target.value)} />
      <TextInput label="Country" value={country} onChange={(e) => setCountry(e.target.value)} />
      {error && <Text c="red" size="xs">{(error as Error).message}</Text>}
      <Button onClick={handleSubmit} loading={isPending}>Run &amp; Submit</Button>
    </Stack>
  );
}
