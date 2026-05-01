"use client";

import type { ReactElement } from "react";
import { Button, FileInput, Stack, Text, TextInput } from "@mantine/core";
import { useState } from "react";
import { exportChallengeUrl } from "@/lib/api-client";
import { useSubmitOwnMethod } from "../hooks/use-submissions";

interface Props {
  challengeId: number;
  onSuccess: () => void;
}

export function SubmissionFormUpload({ challengeId, onSuccess }: Props): ReactElement {
  const [file, setFile] = useState<File | null>(null);
  const [organization, setOrganization] = useState("");
  const [country, setCountry] = useState("");
  const [calcMethod, setCalcMethod] = useState("");
  const { mutate, isPending, error } = useSubmitOwnMethod(challengeId);

  function handleSubmit() {
    if (!file) return;
    mutate({ file, meta: { organization, country, calculation_method: calcMethod } }, { onSuccess });
  }

  return (
    <Stack gap="sm">
      <Text size="sm" c="dimmed">
        Download the test data, calculate 305-day yields with your own method, then upload your
        results as a CSV with columns: <code>cow_id, yield_305day</code>.
      </Text>
      <Button
        variant="outline"
        component="a"
        href={exportChallengeUrl(challengeId)}
        download
      >
        Download test data CSV
      </Button>
      <FileInput
        label="Upload results CSV"
        accept=".csv"
        value={file}
        onChange={setFile}
        placeholder="challenge_results.csv"
      />
      <TextInput label="Organization" value={organization} onChange={(e) => setOrganization(e.target.value)} />
      <TextInput label="Country" value={country} onChange={(e) => setCountry(e.target.value)} />
      <TextInput label="Calculation method" value={calcMethod} onChange={(e) => setCalcMethod(e.target.value)} />
      {error && <Text c="red" size="xs">{(error as Error).message}</Text>}
      <Button onClick={handleSubmit} loading={isPending} disabled={!file}>Submit</Button>
    </Stack>
  );
}
