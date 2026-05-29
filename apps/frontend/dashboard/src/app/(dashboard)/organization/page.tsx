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
  Modal,
  Select,
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
  updateOrganizationMemberRole,
  updateOrganization,
} from "@/lib/api-client";
import type { OrganizationMemberRead, OrganizationMemberRole } from "@/lib/api-client";
import { useAuth } from "@/lib/auth";
import { CenteredLoader } from "@/components/dashboard/centered-loader";

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
  const [inviteRole, setInviteRole] = useState<OrganizationMemberRole>("Member");
  const [memberToRemove, setMemberToRemove] = useState<OrganizationMemberRead | null>(null);
  const [pendingRoleChange, setPendingRoleChange] = useState<{
    member: OrganizationMemberRead;
    role: OrganizationMemberRole;
  } | null>(null);
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
    mutationFn: () => createOrganizationInvite(organizationId ?? 0, inviteRole),
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
      setMemberToRemove(null);
    },
  });
  const updateMemberRoleMutation = useMutation({
    mutationFn: ({ userId, role }: { userId: number; role: OrganizationMemberRole }) =>
      updateOrganizationMemberRole(organizationId ?? 0, userId, role),
    onSuccess: () => {
      if (organizationId) {
        qc.invalidateQueries({ queryKey: membersKey(organizationId) });
      }
      setPendingRoleChange(null);
    },
  });

  if (!user) return <CenteredLoader label="Loading organization..." />;

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

      {canManage && (
        <Card withBorder radius="sm" padding="md">
          <Stack gap="sm">
            <Text fw={600}>Name</Text>
            <Group align="flex-end">
              <TextInput
                aria-label="Organization name"
                value={name}
                onChange={(event) => setName(event.currentTarget.value)}
              />
              <Button
                onClick={() => renameMutation.mutate()}
                disabled={name.trim().length === 0}
                loading={renameMutation.isPending}
              >
                Save
              </Button>
            </Group>
          </Stack>
        </Card>
      )}

      {user.is_admin && (
        <Alert color="green" variant="light">
          Global Admin access is managed through Microsoft Entra app role assignments, not through
          organization invites.
        </Alert>
      )}

      {canManage && (
        <Card withBorder radius="sm" padding="md">
          <Stack gap="md">
            <Group justify="space-between">
              <Text fw={600}>Invite links</Text>
              <Group gap="xs" align="flex-end">
                <Select
                  aria-label="Invite role"
                  label="Invite as"
                  size="xs"
                  value={inviteRole}
                  onChange={(value) => setInviteRole((value as OrganizationMemberRole) ?? "Member")}
                  data={[
                    { value: "Member", label: "Member" },
                    { value: "Owner", label: "Owner" },
                  ]}
                />
                <Button
                  leftSection={<LinkIcon size={14} />}
                  onClick={() => createInviteMutation.mutate()}
                  loading={createInviteMutation.isPending}
                >
                  Create invite
                </Button>
              </Group>
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
                    <Table.Th>Role</Table.Th>
                    <Table.Th>Expires</Table.Th>
                    <Table.Th>Accepted</Table.Th>
                    <Table.Th />
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {(invites.data ?? []).map((invite) => (
                    <Table.Tr key={invite.id}>
                      <Table.Td>{formatDate(invite.created_at)}</Table.Td>
                      <Table.Td>{invite.role}</Table.Td>
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
                          <Select
                            aria-label={`Role for ${member.name ?? member.email ?? member.user_id}`}
                            size="xs"
                            value={member.role}
                            onChange={(value) => {
                              const role = (value as OrganizationMemberRole) ?? "Member";
                              if (role !== member.role) {
                                setPendingRoleChange({ member, role });
                              }
                            }}
                            data={[
                              { value: "Member", label: "Member" },
                              { value: "Owner", label: "Owner" },
                            ]}
                            disabled={
                              member.user_id === user.id || updateMemberRoleMutation.isPending
                            }
                          />
                          <Button
                            size="xs"
                            variant="subtle"
                            color="red"
                            leftSection={<Trash2 size={12} />}
                            onClick={() => setMemberToRemove(member)}
                            disabled={member.user_id === user.id}
                            loading={
                              removeMemberMutation.isPending &&
                              removeMemberMutation.variables === member.user_id
                            }
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

      <Modal
        opened={memberToRemove !== null}
        onClose={() => setMemberToRemove(null)}
        title="Remove member"
        centered
      >
        <Stack gap="md">
          <Text size="sm">
            Are you sure you want to remove{" "}
            <strong>
              {memberToRemove?.name ?? memberToRemove?.email ?? `User #${memberToRemove?.user_id}`}
            </strong>{" "}
            from this organization?
          </Text>
          <Group justify="flex-end">
            <Button variant="subtle" onClick={() => setMemberToRemove(null)}>
              Cancel
            </Button>
            <Button
              color="red"
              loading={removeMemberMutation.isPending}
              onClick={() => {
                if (memberToRemove) {
                  removeMemberMutation.mutate(memberToRemove.user_id);
                }
              }}
            >
              Remove member
            </Button>
          </Group>
        </Stack>
      </Modal>

      <Modal
        opened={pendingRoleChange !== null}
        onClose={() => setPendingRoleChange(null)}
        title={pendingRoleChange?.role === "Owner" ? "Promote member" : "Change member role"}
        centered
      >
        <Stack gap="md">
          <Text size="sm">
            Are you sure you want to change{" "}
            <strong>
              {pendingRoleChange?.member.name ??
                pendingRoleChange?.member.email ??
                `User #${pendingRoleChange?.member.user_id}`}
            </strong>{" "}
            to <strong>{pendingRoleChange?.role}</strong>?
          </Text>
          <Group justify="flex-end">
            <Button variant="subtle" onClick={() => setPendingRoleChange(null)}>
              Cancel
            </Button>
            <Button
              loading={updateMemberRoleMutation.isPending}
              onClick={() => {
                if (pendingRoleChange) {
                  updateMemberRoleMutation.mutate({
                    userId: pendingRoleChange.member.user_id,
                    role: pendingRoleChange.role,
                  });
                }
              }}
            >
              Confirm role change
            </Button>
          </Group>
        </Stack>
      </Modal>
    </div>
  );
}
