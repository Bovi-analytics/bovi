"use client";

import type { ReactElement } from "react";
import Image from "next/image";
import Link from "next/link";
import { Alert, Badge, Button, Group, Paper, Stack, Text, Title } from "@mantine/core";
import { ArrowLeft, BarChart3, Building2, LineChart, LogIn, ShieldCheck } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { startLogin, useAuth } from "@/lib/auth";

function getNextPath(): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("next");
}

export default function LoginPage(): ReactElement {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();
  const [nextPath, setNextPath] = useState<string | null>(null);

  useEffect(() => {
    setNextPath(getNextPath());
  }, []);

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      router.push(nextPath ?? "/");
    }
  }, [isAuthenticated, isLoading, nextPath, router]);

  return (
    <main className="min-h-screen bg-background px-6 py-8 text-foreground">
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-6xl flex-col">
        <Group justify="space-between" align="center">
          <Link href="/" aria-label="Back to Bovi Analytics home">
            <Image
              src="/bovi-logo.png"
              alt="Bovi-Analytics"
              width={2255}
              height={699}
              priority
              className="h-auto w-[150px]"
            />
          </Link>
          <Button component={Link} href="/" variant="subtle" leftSection={<ArrowLeft size={15} />}>
            Home
          </Button>
        </Group>

        <div className="grid flex-1 items-center gap-8 py-10 lg:grid-cols-[1.15fr_0.85fr]">
          <Stack gap="xl">
            <Stack gap="md">
              <Badge variant="light" color="blue" w="fit-content">
                Bovi Analytics dashboard
              </Badge>
              <Title order={1} className="max-w-3xl">
                Sign in to analyze lactation curves for your organization
              </Title>
              <Text size="lg" c="dimmed" maw={680}>
                Fit classical and AI lactation models, compare curve characteristics, manage herd
                profiles, and benchmark milk-yield calculations from a protected organization
                workspace.
              </Text>
            </Stack>

            <div className="grid gap-3 sm:grid-cols-3">
              {[
                {
                  icon: LineChart,
                  title: "Curves",
                  text: "Visualize milk yield over days in milk.",
                },
                {
                  icon: BarChart3,
                  title: "Benchmark",
                  text: "Validate cumulative lactation yield methods.",
                },
                {
                  icon: Building2,
                  title: "Organizations",
                  text: "Work inside the right farm or team context.",
                },
              ].map((item) => {
                const Icon = item.icon;
                return (
                  <Paper key={item.title} withBorder radius="sm" p="md" bg="transparent">
                    <Stack gap="xs">
                      <Icon size={22} className="text-primary" />
                      <Text fw={700}>{item.title}</Text>
                      <Text size="sm" c="dimmed">
                        {item.text}
                      </Text>
                    </Stack>
                  </Paper>
                );
              })}
            </div>
          </Stack>

          <Paper withBorder radius="md" p="xl" className="bg-card/90">
            <Stack gap="lg">
              <Stack gap={4}>
                <Title order={2}>Organization sign-in</Title>
                <Text c="dimmed" size="sm">
                  Use your Microsoft Entra ID account to open the Bovi workspace linked to your
                  organizations.
                </Text>
              </Stack>

              <Alert color="blue" variant="light" icon={<ShieldCheck size={18} />}>
                Your accessible organizations and role are loaded after sign-in. Admins can switch
                across organizations; members can switch between their own organizations.
              </Alert>

              <Button
                size="md"
                leftSection={<LogIn size={16} />}
                onClick={() => void startLogin(nextPath)}
                loading={isLoading}
              >
                Sign in with Microsoft
              </Button>

              <Text size="xs" c="dimmed">
                After sign-in you will see your user identity, active organization, and role in the
                dashboard navigation.
              </Text>
            </Stack>
          </Paper>
        </div>
      </div>
    </main>
  );
}
