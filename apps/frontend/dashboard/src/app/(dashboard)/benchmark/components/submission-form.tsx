"use client";

import type { ReactElement } from "react";
import { Tabs } from "@mantine/core";
import { SubmissionFormBovi } from "./submission-form-bovi";
import { SubmissionFormUpload } from "./submission-form-upload";

interface Props {
  challengeId: number;
  onSuccess: () => void;
}

export function SubmissionForm({ challengeId, onSuccess }: Props): ReactElement {
  return (
    <Tabs defaultValue="bovi">
      <Tabs.List>
        <Tabs.Tab value="bovi">Bovi model</Tabs.Tab>
        <Tabs.Tab value="upload">Own method (CSV)</Tabs.Tab>
      </Tabs.List>
      <Tabs.Panel value="bovi" pt="sm">
        <SubmissionFormBovi challengeId={challengeId} onSuccess={onSuccess} />
      </Tabs.Panel>
      <Tabs.Panel value="upload" pt="sm">
        <SubmissionFormUpload challengeId={challengeId} onSuccess={onSuccess} />
      </Tabs.Panel>
    </Tabs>
  );
}
