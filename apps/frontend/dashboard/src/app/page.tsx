"use client";

import type { ReactElement } from "react";
import Image from "next/image";
import Link from "next/link";
import { Button, Group, Paper, Stack, Text, Title } from "@mantine/core";
import {
  ArrowDown,
  ClipboardList,
  ChevronRight,
  Database,
  FlaskConical,
  Sparkles,
  Trophy,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { DashboardShell } from "@/components/dashboard/dashboard-shell";
import { useAuth } from "@/lib/auth";

interface FlowStep {
  number: string;
  icon: LucideIcon;
  title: string;
  tab: string;
  href: string;
  cta: string;
  description: string;
  sparkle?: boolean;
}

const FLOW: FlowStep[] = [
  {
    number: "1",
    icon: Database,
    title: "Upload or select data",
    tab: "Data Upload tab",
    href: "/data-upload",
    cta: "Load data",
    description:
      "Choose the data you would like to work with. Upload your own milk recording data or explore the platform using one of the built-in demo herds.",
  },
  {
    number: "2",
    icon: ClipboardList,
    title: "Create herd profiles",
    tab: "Herd Profiles tab",
    href: "/herd-profiles",
    cta: "Open Herd Profiles",
    description:
      "Optional: For the best performance with the AI autoencoder model, set the herd statistics for your uploaded data. You can enter these values manually or let the platform calculate them automatically. Save your herd profile for future analyses and faster setup.",
  },
  {
    number: "3",
    icon: FlaskConical,
    title: "Analyze lactation curves",
    tab: "Curves tab",
    href: "/curves",
    cta: "Open Curves",
    sparkle: true,
    description:
      "Experiment with different lactation curve models on individual lactations, calculate key curve characteristics, and visualize the lactation shape of your selected lactations.",
  },
  {
    number: "4",
    icon: Trophy,
    title: "Benchmark a calculation",
    tab: "Benchmark tab",
    href: "/benchmark",
    cta: "Open Benchmark",
    sparkle: true,
    description:
      "Validate and benchmark your cumulative lactation yield calculations with ease. Download a sample dataset, apply your own methods, and upload the results for instant evaluation. Compare your estimates against actual milk yield records or against built-in reference calculations. Explore detailed performance metrics, parity-specific breakdowns, and download a comprehensive PDF report. You can also compare built-in calculation methods side by side to discover which approach performs best for your herd or application.",
  },
];

function HomeContent({ isAuthenticated }: { readonly isAuthenticated: boolean }): ReactElement {
  return (
    <div className="relative flex min-h-full flex-col items-center py-12">
      {/* Decorative gradient background */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-[420px] bg-gradient-to-b from-primary/15 via-primary/5 to-transparent"
      />

      <Stack align="center" gap="xl" maw={920} w="100%" style={{ position: "relative" }}>
        {/* Hero */}
        <Stack align="center" gap="md" w="100%">
          <Image
            src="/bovi-logo.png"
            alt="Bovi-Analytics"
            width={2255}
            height={699}
            priority
            className="h-auto w-full max-w-[280px] drop-shadow-[0_0_24px_hsl(var(--primary)/0.35)]"
          />
          <Title order={1} ta="center">
            From raw records to validated lactation curves
          </Title>
          <Text size="md" ta="center" maw={640}>
            Bovi Analytics helps dairy teams explore, fit, and benchmark lactation curves from
            browser-based herd records.
            {!isAuthenticated && " Sign in to open your organization workspace."}
          </Text>
          {!isAuthenticated && (
            <Group gap="sm" mt="xs">
              <Button component={Link} href="/auth/login" rightSection={<ChevronRight size={16} />}>
                Create account
              </Button>
            </Group>
          )}
        </Stack>

        {/* Flow steps with arrow connectors */}
        <Stack gap={0} w="100%" align="stretch">
          {FLOW.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div key={step.number}>
                <Paper
                  withBorder
                  radius="md"
                  p={0}
                  style={{ overflow: "hidden", position: "relative" }}
                >
                  {/* Left accent bar */}
                  <div
                    aria-hidden
                    className="absolute left-0 top-0 h-full w-1 bg-gradient-to-b from-primary to-accent"
                  />

                  <Group align="stretch" gap={0} wrap="nowrap" className="relative">
                    {/* Step number + icon column - fixed width for consistency */}
                    <Stack
                      align="center"
                      justify="center"
                      gap="sm"
                      py="lg"
                      style={{
                        width: 140,
                        minWidth: 140,
                        flexShrink: 0,
                        background:
                          "linear-gradient(180deg, hsl(var(--primary)/0.10), hsl(var(--accent)/0.05))",
                      }}
                    >
                      <Text size="xs" fw={700} c="blue.4" style={{ letterSpacing: "0.1em" }}>
                        STEP {step.number}
                      </Text>
                      <div className="flex h-16 w-16 items-center justify-center rounded-xl border border-primary/40 bg-primary/15 shadow-[0_0_24px_hsl(var(--primary)/0.25)]">
                        <Icon size={32} className="text-primary" />
                      </div>
                    </Stack>

                    {/* Body */}
                    <Stack gap="xs" p="lg" style={{ flex: 1 }}>
                      <Group gap="xs" align="center" wrap="wrap">
                        <Title order={3} fw={700}>
                          {step.title}
                        </Title>
                        {step.sparkle && (
                          <Sparkles
                            size={18}
                            className="text-accent"
                            style={{
                              filter: "drop-shadow(0 0 6px hsl(var(--accent)/0.6))",
                            }}
                          />
                        )}
                      </Group>
                      <Text size="xs" c="blue.4" fw={600}>
                        {step.tab}
                      </Text>
                      <Text size="sm" mt={4}>
                        {step.description}
                      </Text>
                      {isAuthenticated && (
                        <Group mt="sm">
                          <Button
                            component={Link}
                            href={step.href}
                            size="sm"
                            variant="light"
                            rightSection={<ChevronRight size={14} />}
                          >
                            {step.cta}
                          </Button>
                        </Group>
                      )}
                    </Stack>
                  </Group>
                </Paper>

                {/* Arrow between cards */}
                {idx < FLOW.length - 1 && (
                  <div className="flex justify-center py-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-full border border-primary/40 bg-primary/15 shadow-[0_0_18px_hsl(var(--primary)/0.35)]">
                      <ArrowDown size={18} className="text-primary" />
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </Stack>

        {/* Final CTA - the logical first action after reading the flow */}
        <Stack align="center" gap="xs" mt="md">
          <Text size="sm" ta="center">
            {isAuthenticated
              ? "Ready to follow the flow? Start at step 1."
              : "The workflow opens after sign-in so your data stays tied to the right organization."}
          </Text>
          <Button
            component={Link}
            href={isAuthenticated ? "/data-upload" : "/auth/login"}
            size="lg"
            rightSection={<ChevronRight size={18} />}
          >
            {isAuthenticated ? "Get Started: Go to Data Upload" : "Get Started: Create Account"}
          </Button>
        </Stack>
      </Stack>
    </div>
  );
}

export default function HomePage(): ReactElement {
  const { isAuthenticated } = useAuth();

  return (
    <DashboardShell>
      <HomeContent isAuthenticated={isAuthenticated} />
    </DashboardShell>
  );
}
