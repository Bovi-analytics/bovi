"use client";

import type { ReactElement } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Avatar, Badge, Button, Group, Select, Stack, Text } from "@mantine/core";
import { LogIn, LogOut } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";
import { DASHBOARD_NAVIGATION } from "./navigation";
import {
  getSelectedOrganizationLabel,
  getUserDisplayName,
  getUserInitials,
} from "./sidebar-identity";

export function Sidebar(): ReactElement {
  const pathname = usePathname();
  const { logout, user, selectedOrganizationId, setSelectedOrganizationId } = useAuth();
  const organizationOptions = [
    ...(user?.is_admin ? [{ value: "all", label: "All organizations" }] : []),
    ...(user?.organizations.map((org) => ({ value: String(org.id), label: org.name })) ?? []),
  ];
  const selectedOrganization =
    typeof selectedOrganizationId === "number"
      ? user?.organizations.find((org) => org.id === selectedOrganizationId)
      : null;
  const selectedOrganizationLabel = user
    ? getSelectedOrganizationLabel(selectedOrganizationId, user.organizations)
    : "No organization selected";

  return (
    <aside className="sticky top-0 hidden h-screen w-52 shrink-0 border-r border-border/40 bg-card/80 p-3 text-sm text-muted-foreground md:flex md:flex-col">
      {/* Header */}
      <div className="flex flex-col gap-3 px-3 py-2">
        <Image
          src="/bovi-logo.png"
          alt="Bovi-Analytics"
          width={2255}
          height={699}
          priority
          className="h-auto w-full max-w-[160px]"
        />
        <h2 className="text-base font-semibold text-foreground">Lactation Curves</h2>
        {user && (
          <Stack gap={8} className="rounded-md border border-border/50 bg-background/50 p-2">
            <Group gap="xs" align="center" wrap="nowrap">
              <Avatar size={34} radius="xl" color="blue">
                {getUserInitials(user)}
              </Avatar>
              <div className="min-w-0">
                <Text size="xs" fw={700} className="truncate text-foreground">
                  {getUserDisplayName(user)}
                </Text>
                {user.email && (
                  <Text size="xs" c="dimmed" className="truncate">
                    {user.email}
                  </Text>
                )}
              </div>
            </Group>
            <Group gap={6}>
              {user.is_admin && (
                <Badge size="xs" variant="light" color="green">
                  Admin
                </Badge>
              )}
              {selectedOrganization?.role && (
                <Badge size="xs" variant="light" color="blue">
                  {selectedOrganization.role}
                </Badge>
              )}
            </Group>
            <Text size="xs" c="dimmed" className="line-clamp-2">
              {selectedOrganizationLabel}
            </Text>
          </Stack>
        )}
        {organizationOptions.length > 0 && (
          <Select
            aria-label="Organization"
            size="xs"
            data={organizationOptions}
            value={selectedOrganizationId === null ? null : String(selectedOrganizationId)}
            onChange={(value) => {
              if (value === "all") {
                setSelectedOrganizationId("all");
              } else if (value) {
                setSelectedOrganizationId(Number.parseInt(value, 10));
              }
            }}
            comboboxProps={{ withinPortal: false }}
          />
        )}
        {user && (
          <Button
            size="xs"
            variant="subtle"
            color="gray"
            leftSection={<LogOut size={13} />}
            onClick={() => void logout()}
          >
            Sign out
          </Button>
        )}
        {!user && (
          <Button
            component={Link}
            href="/auth/login"
            size="xs"
            variant="light"
            leftSection={<LogIn size={13} />}
          >
            Sign in
          </Button>
        )}
      </div>

      {/* Navigation */}
      <nav className="mt-6 flex flex-col gap-1">
        {DASHBOARD_NAVIGATION.map((item) => {
          const isActive = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 transition-colors",
                isActive ? "bg-primary/10 text-primary" : "hover:bg-muted/40 hover:text-foreground"
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span className="font-medium tracking-wide">{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
