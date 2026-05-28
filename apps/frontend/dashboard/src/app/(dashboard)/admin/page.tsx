"use client";

import type { ReactElement } from "react";
import { useMemo, useState } from "react";
import Link from "next/link";
import {
  Alert,
  Badge,
  Button,
  Group,
  Loader,
  Modal,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Tabs,
  Text,
  TextInput,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useQuery } from "@tanstack/react-query";
import { Building2, Download, ExternalLink, Layers3, Search, ShieldCheck } from "lucide-react";
import {
  adminOverviewOptionsKey,
  downloadSubmissionReport,
  listAdminSubmissionsOverview,
  listOrganizations,
  type AdminOverviewOptions,
} from "@/lib/api-client";
import { useAuth } from "@/lib/auth";
import type {
  AdminCategoryBreakdown,
  AdminDataCategory,
  AdminOrganizationBreakdown,
  AdminOverviewItem,
} from "@/types/api";

const CATEGORY_OPTIONS: { label: string; value: AdminDataCategory | "all" }[] = [
  { label: "All resources", value: "all" },
  { label: "Benchmark submissions", value: "benchmark_submission" },
  { label: "Benchmark challenges", value: "benchmark_challenge" },
  { label: "Herd dataset uploads", value: "herd_dataset_upload" },
  { label: "Herd profiles", value: "herd_profile" },
];

function formatDate(value: string | null | undefined): string {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function formatNumber(value: number | null | undefined): string {
  return value === null || value === undefined ? "-" : value.toLocaleString();
}

function statusColor(status: string, failedCount: number): string {
  if (failedCount > 0) return "orange";
  if (status === "ready" || status === "completed") return "green";
  if (status === "warning") return "yellow";
  return "red";
}

function categoryColor(category: AdminDataCategory): string {
  return {
    benchmark_submission: "blue",
    benchmark_challenge: "cyan",
    herd_dataset_upload: "grape",
    herd_profile: "teal",
  }[category];
}

function itemAction(item: AdminOverviewItem, onDetails: (item: AdminOverviewItem) => void) {
  if (item.item_type === "benchmark_challenge" && item.numeric_id) {
    return (
      <Button
        component={Link}
        href={`/benchmark/${item.numeric_id}`}
        size="xs"
        variant="light"
        rightSection={<ExternalLink size={12} />}
      >
        Open
      </Button>
    );
  }
  if (item.item_type === "benchmark_submission" && item.numeric_id) {
    return (
      <Group gap="xs" wrap="nowrap">
        {item.challenge_id && (
          <Button
            component={Link}
            href={`/benchmark/${item.challenge_id}`}
            size="xs"
            variant="light"
            rightSection={<ExternalLink size={12} />}
          >
            Context
          </Button>
        )}
        <Button
          size="xs"
          variant="subtle"
          leftSection={<Download size={12} />}
          onClick={() => void downloadSubmissionReport(item.numeric_id ?? 0)}
        >
          Report
        </Button>
      </Group>
    );
  }
  return (
    <Button size="xs" variant="light" onClick={() => onDetails(item)}>
      Details
    </Button>
  );
}

function KpiTile({
  label,
  value,
  tone = "default",
}: {
  readonly label: string;
  readonly value: number | string;
  readonly tone?: "default" | "warning";
}): ReactElement {
  return (
    <Paper withBorder radius="sm" p="md">
      <Text size="xs" c="dimmed" fw={700}>
        {label}
      </Text>
      <Text size="xl" fw={800} c={tone === "warning" ? "orange" : undefined}>
        {typeof value === "number" ? value.toLocaleString() : value}
      </Text>
    </Paper>
  );
}

function CompanyTable({
  rows,
  dense = false,
}: {
  readonly rows: AdminOrganizationBreakdown[];
  readonly dense?: boolean;
}): ReactElement {
  return (
    <Table striped highlightOnHover fz="sm">
      <Table.Thead>
        <Table.Tr>
          <Table.Th>Company</Table.Th>
          <Table.Th>Users</Table.Th>
          <Table.Th>Total</Table.Th>
          {!dense && <Table.Th>Submissions</Table.Th>}
          {!dense && <Table.Th>Uploads</Table.Th>}
          <Table.Th>Problems</Table.Th>
          <Table.Th>Latest</Table.Th>
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {rows.map((org) => (
          <Table.Tr key={org.organization_id ?? org.organization_name}>
            <Table.Td>{org.organization_name}</Table.Td>
            <Table.Td>{org.user_count}</Table.Td>
            <Table.Td>{org.total_items}</Table.Td>
            {!dense && <Table.Td>{org.benchmark_submissions}</Table.Td>}
            {!dense && <Table.Td>{org.herd_dataset_uploads}</Table.Td>}
            <Table.Td>
              <Badge color={org.failed_items > 0 ? "orange" : "green"} variant="light">
                {org.failed_items}
              </Badge>
            </Table.Td>
            <Table.Td>{formatDate(org.latest_activity_at)}</Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}

function CategoryTable({ rows }: { readonly rows: AdminCategoryBreakdown[] }): ReactElement {
  return (
    <Table striped highlightOnHover fz="sm">
      <Table.Thead>
        <Table.Tr>
          <Table.Th>Resource</Table.Th>
          <Table.Th>Count</Table.Th>
          <Table.Th>Problems</Table.Th>
          <Table.Th>Latest</Table.Th>
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {rows.map((cat) => (
          <Table.Tr key={cat.item_type}>
            <Table.Td>
              <Badge color={categoryColor(cat.item_type)} variant="light">
                {cat.label}
              </Badge>
            </Table.Td>
            <Table.Td>{cat.count}</Table.Td>
            <Table.Td>{cat.failed_count}</Table.Td>
            <Table.Td>{formatDate(cat.latest_activity_at)}</Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}

export default function AdminPage(): ReactElement {
  const { user } = useAuth();
  const [organizationId, setOrganizationId] = useState<number | "all">("all");
  const [category, setCategory] = useState<AdminDataCategory | "all">("all");
  const [q, setQ] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [sort, setSort] = useState<NonNullable<AdminOverviewOptions["sort"]>>("created_at");
  const [direction, setDirection] =
    useState<NonNullable<AdminOverviewOptions["direction"]>>("desc");
  const [selectedItem, setSelectedItem] = useState<AdminOverviewItem | null>(null);
  const [detailsOpen, detailsHandlers] = useDisclosure(false);

  const activityOptions = useMemo<AdminOverviewOptions>(
    () => ({
      organizationId,
      category,
      q: q.trim() || undefined,
      from: from || undefined,
      to: to || undefined,
      sort,
      direction,
      limit: 100,
    }),
    [category, direction, from, organizationId, q, sort, to]
  );

  const organizationsQuery = useQuery({
    queryKey: ["admin-organizations"],
    queryFn: listOrganizations,
    enabled: Boolean(user?.is_admin),
  });
  const homeQuery = useQuery({
    queryKey: ["admin-submissions-overview", "home"],
    queryFn: () => listAdminSubmissionsOverview({ organizationId: "all", limit: 12 }),
    enabled: Boolean(user?.is_admin),
  });
  const activityQuery = useQuery({
    queryKey: ["admin-submissions-overview", "activity", adminOverviewOptionsKey(activityOptions)],
    queryFn: () => listAdminSubmissionsOverview(activityOptions),
    enabled: Boolean(user?.is_admin),
  });

  const organizationOptions = [
    { label: "All companies", value: "all" },
    ...(organizationsQuery.data?.map((org) => ({ label: org.name, value: String(org.id) })) ?? []),
  ];

  const openDetails = (item: AdminOverviewItem) => {
    setSelectedItem(item);
    detailsHandlers.open();
  };

  if (!user?.is_admin) {
    return (
      <Alert color="red" variant="light" title="Admin access required">
        This page is only available to global Bovi admins.
      </Alert>
    );
  }

  if (homeQuery.isLoading) return <Loader />;
  if (homeQuery.error) return <Text c="red">Failed to load admin overview.</Text>;

  const home = homeQuery.data;
  const activity = activityQuery.data;

  return (
    <div className="space-y-6 p-6">
      <Group justify="space-between" align="flex-start">
        <Stack gap={2}>
          <Group gap="xs">
            <ShieldCheck size={22} className="text-primary" />
            <h1 className="text-2xl font-semibold">Admin</h1>
          </Group>
          <Text size="sm" c="dimmed">
            Cross-company view of submissions, uploads, challenges, and herd profiles.
          </Text>
        </Stack>
        <Badge color="blue" variant="light">
          Read-only
        </Badge>
      </Group>

      <Tabs defaultValue="home" keepMounted={false}>
        <Tabs.List>
          <Tabs.Tab value="home" leftSection={<ShieldCheck size={14} />}>
            Home
          </Tabs.Tab>
          <Tabs.Tab value="activity" leftSection={<Search size={14} />}>
            Activity
          </Tabs.Tab>
          <Tabs.Tab value="companies" leftSection={<Building2 size={14} />}>
            Companies
          </Tabs.Tab>
          <Tabs.Tab value="resources" leftSection={<Layers3 size={14} />}>
            Resources
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="home" pt="md">
          {home && (
            <Stack gap="md">
              <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
                <KpiTile label="Total items" value={home.kpis.total_items} />
                <KpiTile label="Companies" value={home.kpis.organizations} />
                <KpiTile label="Users" value={home.kpis.users} />
                <KpiTile label="Problem signals" value={home.kpis.failed_items} tone="warning" />
              </SimpleGrid>
              <SimpleGrid cols={{ base: 1, lg: 2 }}>
                <Paper withBorder radius="sm" p="md">
                  <Group justify="space-between" mb="sm">
                    <Text fw={700} size="sm">
                      Company snapshot
                    </Text>
                    <Text size="xs" c="dimmed">
                      Top recent companies
                    </Text>
                  </Group>
                  <CompanyTable rows={home.by_organization.slice(0, 6)} dense />
                </Paper>
                <Paper withBorder radius="sm" p="md">
                  <Text fw={700} size="sm" mb="sm">
                    Resource mix
                  </Text>
                  <CategoryTable rows={home.by_category} />
                </Paper>
              </SimpleGrid>
              <Paper withBorder radius="sm" p="md">
                <Group justify="space-between" mb="sm">
                  <Text fw={700} size="sm">
                    Latest activity
                  </Text>
                  <Text size="xs" c="dimmed">
                    Unfiltered
                  </Text>
                </Group>
                <ActivityTable items={home.items.slice(0, 12)} onDetails={openDetails} />
              </Paper>
            </Stack>
          )}
        </Tabs.Panel>

        <Tabs.Panel value="activity" pt="md">
          <Stack gap="md">
            <Paper withBorder radius="sm" p="md">
              <Group gap="sm" align="flex-end">
                <Select
                  label="Company"
                  size="xs"
                  data={organizationOptions}
                  value={String(organizationId)}
                  onChange={(value) => {
                    if (!value || value === "all") {
                      setOrganizationId("all");
                    } else {
                      setOrganizationId(Number(value));
                    }
                  }}
                  w={220}
                />
                <Select
                  label="Resource"
                  size="xs"
                  data={CATEGORY_OPTIONS}
                  value={category}
                  onChange={(value) => setCategory((value as AdminDataCategory | "all") ?? "all")}
                  w={220}
                />
                <TextInput
                  label="Search"
                  size="xs"
                  leftSection={<Search size={14} />}
                  value={q}
                  onChange={(event) => setQ(event.currentTarget.value)}
                  placeholder="Company, user, method"
                  w={240}
                />
                <TextInput
                  label="From"
                  size="xs"
                  type="date"
                  value={from}
                  onChange={(event) => setFrom(event.currentTarget.value)}
                />
                <TextInput
                  label="To"
                  size="xs"
                  type="date"
                  value={to}
                  onChange={(event) => setTo(event.currentTarget.value)}
                />
                <Select
                  label="Sort"
                  size="xs"
                  data={[
                    { label: "Created", value: "created_at" },
                    { label: "Company", value: "organization" },
                    { label: "User", value: "user" },
                    { label: "Resource", value: "category" },
                    { label: "Status", value: "status" },
                  ]}
                  value={sort}
                  onChange={(value) =>
                    setSort((value as NonNullable<AdminOverviewOptions["sort"]>) ?? "created_at")
                  }
                  w={150}
                />
                <Select
                  label="Direction"
                  size="xs"
                  data={[
                    { label: "Descending", value: "desc" },
                    { label: "Ascending", value: "asc" },
                  ]}
                  value={direction}
                  onChange={(value) =>
                    setDirection(
                      (value as NonNullable<AdminOverviewOptions["direction"]>) ?? "desc"
                    )
                  }
                  w={140}
                />
              </Group>
            </Paper>
            <Paper withBorder radius="sm" p="md">
              {activityQuery.isLoading && <Loader />}
              {activityQuery.error && <Text c="red">Failed to load activity.</Text>}
              {activity && <ActivityTable items={activity.items} onDetails={openDetails} />}
              {activity?.items.length === 0 && (
                <Text size="sm" c="dimmed" mt="md">
                  No admin records match the selected filters.
                </Text>
              )}
            </Paper>
          </Stack>
        </Tabs.Panel>

        <Tabs.Panel value="companies" pt="md">
          <Paper withBorder radius="sm" p="md">
            {home && <CompanyTable rows={home.by_organization} />}
          </Paper>
        </Tabs.Panel>

        <Tabs.Panel value="resources" pt="md">
          <Paper withBorder radius="sm" p="md">
            {home && <CategoryTable rows={home.by_category} />}
          </Paper>
        </Tabs.Panel>
      </Tabs>

      <Modal opened={detailsOpen} onClose={detailsHandlers.close} title="Submission metadata">
        {selectedItem && (
          <Stack gap="xs">
            <Text size="sm">
              <strong>Name:</strong> {selectedItem.title}
            </Text>
            <Text size="sm">
              <strong>Company:</strong> {selectedItem.organization_name ?? "-"}
            </Text>
            <Text size="sm">
              <strong>User:</strong> {selectedItem.user_email ?? selectedItem.user_name ?? "-"}
            </Text>
            <Text size="sm">
              <strong>Resource:</strong> {selectedItem.item_type_label}
            </Text>
            <Text size="sm">
              <strong>Source:</strong> {selectedItem.source ?? "-"}
            </Text>
            <Text size="sm">
              <strong>Rows:</strong> {formatNumber(selectedItem.row_count)}
            </Text>
            <Text size="sm">
              <strong>Cows:</strong> {formatNumber(selectedItem.cow_count)}
            </Text>
            <Text size="sm">
              <strong>Status:</strong> {selectedItem.status}
            </Text>
          </Stack>
        )}
      </Modal>
    </div>
  );
}

function ActivityTable({
  items,
  onDetails,
}: {
  readonly items: AdminOverviewItem[];
  readonly onDetails: (item: AdminOverviewItem) => void;
}): ReactElement {
  return (
    <Table striped highlightOnHover withColumnBorders fz="sm">
      <Table.Thead>
        <Table.Tr>
          <Table.Th>Time</Table.Th>
          <Table.Th>Company</Table.Th>
          <Table.Th>User</Table.Th>
          <Table.Th>Resource</Table.Th>
          <Table.Th>Name / type</Table.Th>
          <Table.Th>Source</Table.Th>
          <Table.Th>Counts</Table.Th>
          <Table.Th>Status</Table.Th>
          <Table.Th>Metric</Table.Th>
          <Table.Th>Action</Table.Th>
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {items.map((item) => (
          <Table.Tr key={`${item.item_type}-${item.id}`}>
            <Table.Td>{formatDate(item.created_at)}</Table.Td>
            <Table.Td>{item.organization_name ?? "-"}</Table.Td>
            <Table.Td>
              <Text size="sm" lineClamp={1}>
                {item.user_email ?? item.user_name ?? "-"}
              </Text>
            </Table.Td>
            <Table.Td>
              <Badge size="xs" color={categoryColor(item.item_type)} variant="light">
                {item.item_type_label}
              </Badge>
            </Table.Td>
            <Table.Td>
              <Text size="sm" fw={600} maw={220} lineClamp={1}>
                {item.title}
              </Text>
              <Text size="xs" c="dimmed">
                #{item.id}
              </Text>
            </Table.Td>
            <Table.Td>{item.source ?? "-"}</Table.Td>
            <Table.Td>
              <Text size="xs">Rows {formatNumber(item.row_count)}</Text>
              <Text size="xs">Cows {formatNumber(item.cow_count)}</Text>
            </Table.Td>
            <Table.Td>
              <Badge color={statusColor(item.status, item.failed_count)} variant="light">
                {item.failed_count > 0 ? `${item.status} (${item.failed_count})` : item.status}
              </Badge>
            </Table.Td>
            <Table.Td>
              {item.primary_metric_label &&
              item.primary_metric_value !== null &&
              item.primary_metric_value !== undefined
                ? `${item.primary_metric_label} ${item.primary_metric_value.toFixed(2)}`
                : "-"}
            </Table.Td>
            <Table.Td>{itemAction(item, onDetails)}</Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}
