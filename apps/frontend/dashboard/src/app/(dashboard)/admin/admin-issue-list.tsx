import { Alert, Code, Group, Stack, Text } from "@mantine/core";
import { TriangleAlert } from "lucide-react";
import type { ReactElement } from "react";
import type { AdminOverviewIssue } from "@/types/api";

function issueColor(severity: AdminOverviewIssue["severity"]): string {
  return { info: "blue", warning: "yellow", error: "red" }[severity];
}

export function AdminIssueList({
  issues,
}: {
  readonly issues: AdminOverviewIssue[];
}): ReactElement | null {
  if (issues.length === 0) return null;

  return (
    <Stack gap="xs">
      {issues.map((issue, index) => (
        <Alert
          key={`${issue.title}-${issue.message}-${index}`}
          color={issueColor(issue.severity)}
          icon={<TriangleAlert size={15} />}
          title={issue.title}
          variant="light"
        >
          <Stack gap={4}>
            <Text size="sm">{issue.message}</Text>
            {(issue.affected_count != null || issue.sample_ids.length > 0) && (
              <Group gap="md" align="flex-start">
                {issue.affected_count != null && (
                  <Text size="xs" fw={600}>
                    Affected: {issue.affected_count.toLocaleString()}
                  </Text>
                )}
                {issue.sample_ids.length > 0 && (
                  <Text size="xs">
                    Sample IDs: <Code>{issue.sample_ids.join(", ")}</Code>
                  </Text>
                )}
              </Group>
            )}
          </Stack>
        </Alert>
      ))}
    </Stack>
  );
}
