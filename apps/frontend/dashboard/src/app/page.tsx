"use client";

import type { ReactElement } from "react";
import Image from "next/image";
import Link from "next/link";
import { Badge, Button, Group, Paper, Stack, Text, Title } from "@mantine/core";
import {
  ArrowDown,
  BarChart3,
  ChevronRight,
  Database,
  FlaskConical,
  Sparkles,
  Trophy,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { DashboardShell } from "@/components/dashboard/dashboard-shell";

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
    title: "Load herd data",
    tab: "Herd Data tab",
    href: "/herd-stats",
    cta: "Load data",
    description:
      "Start here. Pick a built-in preset farm (Aurora Ridge or Sunnyside) or upload your own milk-recording CSV. Every other page reads from this dataset.",
  },
  {
    number: "2",
    icon: BarChart3,
    title: "Explore herd statistics",
    tab: "Herd Data tab, scroll down",
    href: "/herd-stats",
    cta: "Review herd stats",
    description:
      "On the same Herd Data page, review the ten aggregate KPIs (achieved milk, days in milk, somatic cell score, days open, ...). Sanity-check the import before moving on.",
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
      "Fit lactation curves on individual cows and compare them side-by-side. Pick the curves you want to evaluate further.",
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
      "Validate a 305-day yield calculation against the ICAR reference (TIM). Either let Bovi run a built-in calculation for you, or upload the results of your own method as a CSV. Get a per-parity report and a downloadable PDF.",
  },
];

export default function HomePage(): ReactElement {
  return (
    <DashboardShell>
      <div className="relative flex min-h-full flex-col items-center py-12">
        {/* Decorative gradient background */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-0 h-[420px] bg-gradient-to-b from-primary/15 via-primary/5 to-transparent"
        />

        <Stack
          align="center"
          gap="xl"
          maw={920}
          w="100%"
          style={{ position: "relative" }}
        >
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
            <Badge
              size="lg"
              variant="light"
              color="blue"
              leftSection={<Sparkles size={12} />}
            >
              Lactation Curve Platform
            </Badge>
            <Title order={1} ta="center">
              From raw records to validated lactation curves
            </Title>
            <Text size="md" ta="center" maw={640}>
              Explore, fit, and benchmark dairy cow lactation curves, all from
              your browser. Follow the four steps below in order.
            </Text>
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

                    <Group
                      align="stretch"
                      gap={0}
                      wrap="nowrap"
                      className="relative"
                    >
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
                        <Text
                          size="xs"
                          fw={700}
                          c="blue.4"
                          style={{ letterSpacing: "0.1em" }}
                        >
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
                                filter:
                                  "drop-shadow(0 0 6px hsl(var(--accent)/0.6))",
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
              Ready to follow the flow? Start at step 1.
            </Text>
            <Button
              component={Link}
              href="/herd-stats"
              size="lg"
              rightSection={<ChevronRight size={18} />}
            >
              Get started: go to Herd Stats
            </Button>
          </Stack>
        </Stack>
      </div>
    </DashboardShell>
  );
}
