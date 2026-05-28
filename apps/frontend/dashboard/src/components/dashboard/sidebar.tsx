"use client";

import type { ReactElement } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Avatar,
  Badge,
  Button,
  Group,
  Menu,
  Select,
  Stack,
  Text,
  UnstyledButton,
} from "@mantine/core";
import { Building2, LogIn, LogOut } from "lucide-react";
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
          <Menu position="bottom-start" width={240} shadow="md" closeOnItemClick={false}>
            <Menu.Target>
              <UnstyledButton
                aria-label="Open user menu"
                className="w-fit rounded-full outline-none ring-primary/40 transition hover:ring-2 focus-visible:ring-2"
              >
                <Avatar size={38} radius="xl" color="blue">
                  {getUserInitials(user)}
                </Avatar>
              </UnstyledButton>
            </Menu.Target>
            <Menu.Dropdown>
              <Stack gap={8} className="p-2">
                <div className="min-w-0">
                  <Text size="sm" fw={700} className="truncate text-foreground">
                    {getUserDisplayName(user)}
                  </Text>
                  {user.email && (
                    <Text size="xs" c="dimmed" className="truncate">
                      {user.email}
                    </Text>
                  )}
                </div>
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
                {selectedOrganizationId !== null && selectedOrganizationId !== "all" && (
                  <Button
                    component={Link}
                    href="/organization"
                    size="xs"
                    variant="light"
                    fullWidth
                    leftSection={<Building2 size={13} />}
                  >
                    View organization
                  </Button>
                )}
                {organizationOptions.length > 0 && (
                  <Select
                    label="Organization"
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
              </Stack>
              <Menu.Divider />
              <Menu.Item
                color="gray"
                leftSection={<LogOut size={14} />}
                onClick={() => void logout()}
              >
                Sign out
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>
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
        {DASHBOARD_NAVIGATION.filter((item) => !item.adminOnly || user?.is_admin).map((item) => {
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
