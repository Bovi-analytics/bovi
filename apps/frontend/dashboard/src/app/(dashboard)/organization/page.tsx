"use client";

import type { ReactElement } from "react";
import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Alert,
  Badge,
  Button,
  Card,
  CopyButton,
  Group,
  Loader,
  Stack,
  Table,
  Text,
  TextInput,
} from "@mantine/core";
import { LinkIcon, Trash2 } from "lucide-react";
import {
  createOrganizationInvite,
  listOrganizationInvites,
  listOrganizationMembers,
  removeOrganizationMember,
  revokeOrganizationInvite,
  updateOrganization,
} from "@/lib/api-client";
import { useAuth } from "@/lib/auth";

const membersKey = (organizationId: number) => ["organization-members", organizationId] as const;
const invitesKey = (organizationId: number) => ["organization-invites", organizationId] as const;

function formatDate(value: string | null): string {
  return value ? new Date(value).toLocaleDateString() : "-";
}

export default function OrganizationPage(): ReactElement {
  const qc = useQueryClient();
  const { selectedOrganizationId, setSelectedOrganizationId, user } = useAuth();
  const selectedOrganization = useMemo(
    () =>
      typeof selectedOrganizationId === "number"
        ? user?.organizations.find((org) => org.id === selectedOrganizationId)
        : null,
    [selectedOrganizationId, user?.organizations]
  );
  const [name, setName] = useState(selectedOrganization?.name ?? "");
  const [inviteUrl, setInviteUrl] = useState<string | null>(null);
  const organizationId = typeof selectedOrganizationId === "number" ? selectedOrganizationId : null;
  const canManage = Boolean(user?.is_admin || selectedOrganization?.role === "Owner");

  useEffect(() => {
    setName(selectedOrganization?.name ?? "");
    setInviteUrl(null);
  }, [selectedOrganization?.id, selectedOrganization?.name]);

  const members = useQuery({
    queryKey: organizationId ? membersKey(organizationId) : ["organization-members"],
    queryFn: () => listOrganizationMembers(organizationId ?? 0),
    enabled: organizationId !== null,
  });
  const invites = useQuery({
    queryKey: organizationId ? invitesKey(organizationId) : ["organization-invites"],
    queryFn: () => listOrganizationInvites(organizationId ?? 0),
    enabled: organizationId !== null && canManage,
  });
  const renameMutation = useMutation({
    mutationFn: () => updateOrganization(organizationId ?? 0, name.trim()),
    onSuccess: (org) => {
      setSelectedOrganizationId(org.id);
    },
  });
  const createInviteMutation = useMutation({
    mutationFn: () => createOrganizationInvite(organizationId ?? 0),
    onSuccess: (invite) => {
      const baseUrl = window.location.origin;
      setInviteUrl(`${baseUrl}/join?invite=${encodeURIComponent(invite.token)}`);
      if (organizationId) {
        qc.invalidateQueries({ queryKey: invitesKey(organizationId) });
      }
    },
  });
  const revokeInviteMutation = useMutation({
    mutationFn: (inviteId: number) => revokeOrganizationInvite(organizationId ?? 0, inviteId),
    onSuccess: () => {
      if (organizationId) {
        qc.invalidateQueries({ queryKey: invitesKey(organizationId) });
      }
    },
  });
  const removeMemberMutation = useMutation({
    mutationFn: (userId: number) => removeOrganizationMember(organizationId ?? 0, userId),
    onSuccess: () => {
      if (organizationId) {
        qc.invalidateQueries({ queryKey: membersKey(organizationId) });
      }
    },
  });

  if (!user) return <Loader />;

  if (organizationId === null) {
    return (
      <div className="space-y-6 p-6">
        <h1 className="text-2xl font-semibold">Organization</h1>
        <Alert color="yellow" variant="light">
          Select a specific organization to manage members and invite links.
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <Group justify="space-between" align="flex-start">
        <Stack gap={2}>
          <h1 className="text-2xl font-semibold">Organization</h1>
          <Text size="sm" c="dimmed">
            {selectedOrganization?.name ?? "Selected organization"}
          </Text>
        </Stack>
        {selectedOrganization?.role && (
          <Badge variant="light" color={canManage ? "green" : "gray"}>
            {user.is_admin ? "Admin" : selectedOrganization.role}
          </Badge>
        )}
      </Group>

      {!canManage && (
        <Alert color="blue" variant="light">
          Organization settings and invite links are available to Owners and Admins.
        </Alert>
      )}

      <Card withBorder radius="sm" padding="md">
        <Stack gap="sm">
          <Text fw={600}>Name</Text>
          <Group align="flex-end">
            <TextInput
              aria-label="Organization name"
              value={name}
              onChange={(event) => setName(event.currentTarget.value)}
              disabled={!canManage}
            />
            <Button
              onClick={() => renameMutation.mutate()}
              disabled={!canManage || name.trim().length === 0}
              loading={renameMutation.isPending}
            >
              Save
            </Button>
          </Group>
        </Stack>
      </Card>

      {canManage && (
        <Card withBorder radius="sm" padding="md">
          <Stack gap="md">
            <Group justify="space-between">
              <Text fw={600}>Invite links</Text>
              <Button
                leftSection={<LinkIcon size={14} />}
                onClick={() => createInviteMutation.mutate()}
                loading={createInviteMutation.isPending}
              >
                Create invite
              </Button>
            </Group>
            {inviteUrl && (
              <Group gap="sm">
                <TextInput
                  aria-label="New invite link"
                  value={inviteUrl}
                  readOnly
                  className="flex-1"
                />
                <CopyButton value={inviteUrl}>
                  {({ copied, copy }) => (
                    <Button variant="light" onClick={copy}>
                      {copied ? "Copied" : "Copy"}
                    </Button>
                  )}
                </CopyButton>
              </Group>
            )}
            {invites.isLoading ? (
              <Loader size="sm" />
            ) : (
              <Table striped highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Created</Table.Th>
                    <Table.Th>Expires</Table.Th>
                    <Table.Th>Accepted</Table.Th>
                    <Table.Th />
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {(invites.data ?? []).map((invite) => (
                    <Table.Tr key={invite.id}>
                      <Table.Td>{formatDate(invite.created_at)}</Table.Td>
                      <Table.Td>{formatDate(invite.expires_at)}</Table.Td>
                      <Table.Td>{invite.accepted_count}</Table.Td>
                      <Table.Td>
                        <Group justify="flex-end">
                          <Button
                            size="xs"
                            variant="subtle"
                            color="red"
                            onClick={() => revokeInviteMutation.mutate(invite.id)}
                          >
                            Revoke
                          </Button>
                        </Group>
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            )}
          </Stack>
        </Card>
      )}

      <Card withBorder radius="sm" padding="md">
        <Stack gap="md">
          <Text fw={600}>Members</Text>
          {members.isLoading ? (
            <Loader size="sm" />
          ) : (
            <Table striped highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>Email</Table.Th>
                  <Table.Th>Role</Table.Th>
                  {canManage && <Table.Th />}
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {(members.data ?? []).map((member) => (
                  <Table.Tr key={member.user_id}>
                    <Table.Td>{member.name ?? "-"}</Table.Td>
                    <Table.Td>{member.email ?? "-"}</Table.Td>
                    <Table.Td>{member.role}</Table.Td>
                    {canManage && (
                      <Table.Td>
                        <Group justify="flex-end">
                          <Button
                            size="xs"
                            variant="subtle"
                            color="red"
                            leftSection={<Trash2 size={12} />}
                            onClick={() => removeMemberMutation.mutate(member.user_id)}
                            disabled={member.user_id === user.id}
                          >
                            Remove
                          </Button>
                        </Group>
                      </Table.Td>
                    )}
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          )}
        </Stack>
      </Card>
    </div>
  );
}
