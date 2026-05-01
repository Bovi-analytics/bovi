"use client";

import type { ReactElement } from "react";
import Link from "next/link";
import { Button, Group, Paper, Stack, Text, Title } from "@mantine/core";
import { BarChart3, ChevronRight, FlaskConical, Trophy, Upload } from "lucide-react";
import { DashboardShell } from "@/components/dashboard/dashboard-shell";

const STEPS = [
  {
    number: "1",
    icon: Upload,
    title: "Load herd data",
    description:
      "Upload your own CSV export (ICAR, DairyCom, or aggregated stats) or pick one of the preset farm datasets.",
  },
  {
    number: "2",
    icon: BarChart3,
    title: "Explore statistics",
    description:
      "View per-herd KPIs — achieved milk yields, days in milk, somatic cell score, and more.",
  },
  {
    number: "3",
    icon: FlaskConical,
    title: "Analyze lactation curves",
    description:
      "Fit classical models (Wood, Wilmink, Ali-Schaeffer) or run the deep-learning autoencoder on individual cows from your herd.",
  },
] as const;

export default function HomePage(): ReactElement {
  return (
    <DashboardShell>
      <div className="flex min-h-full flex-col items-center justify-center py-16">
        <Stack align="center" gap="xl" maw={760} w="100%">
          <Stack align="center" gap="xs">
            <Text size="xs" fw={600} tt="uppercase" lts="0.15em" c="violet">
              Bovi Analytics
            </Text>
            <Title order={1} ta="center">
              Lactation Curve Platform
            </Title>
            <Text size="md" c="dimmed" ta="center" maw={500}>
              Explore, fit, and predict dairy cow lactation curves using classical models and deep
              learning — all from your browser.
            </Text>
          </Stack>

          <Group gap="md" grow w="100%" align="stretch">
            {STEPS.map((step) => {
              const Icon = step.icon;
              return (
                <Paper key={step.number} withBorder p="lg" radius="md">
                  <Stack gap="sm" h="100%">
                    <Group gap="xs">
                      <Text size="sm" fw={700} c="violet">
                        {step.number}
                      </Text>
                      <Icon size={16} className="text-muted-foreground" />
                    </Group>
                    <Text fw={600} size="sm">
                      {step.title}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {step.description}
                    </Text>
                  </Stack>
                </Paper>
              );
            })}
          </Group>

          <Paper withBorder p="lg" radius="md" w="100%">
            <Stack gap="sm">
              <Group gap="xs" align="center">
                <Trophy size={18} className="text-yellow-600" />
                <Text fw={700} size="sm">
                  Benchmark — ICAR-style accreditation
                </Text>
              </Group>
              <Text size="sm" c="dimmed">
                Validate a 305-day milk yield calculation method against an ICAR reference. Generate
                a challenge from a preset dataset (Aurora, Sunnyside, or ICAR), then either let Bovi
                compute yields with one of its built-in models or upload results from your own
                method as a CSV. Bovi compares the two and produces a PDF report with Pearson, RMSE,
                MAE, and MAPE — overall and per parity.
              </Text>
              <Text size="xs" c="dimmed">
                Use this when you need to compare a calculation pipeline against a known reference,
                document accreditation evidence, or evaluate a new model against TIM on a fixed
                cohort.
              </Text>
              <Group gap="sm">
                <Button
                  component={Link}
                  href="/benchmark"
                  size="sm"
                  variant="light"
                  color="yellow"
                  rightSection={<ChevronRight size={14} />}
                >
                  Open benchmark
                </Button>
              </Group>
            </Stack>
          </Paper>

          <Button
            component={Link}
            href="/herd-stats"
            size="md"
            color="violet"
            rightSection={<ChevronRight size={16} />}
          >
            Get started
          </Button>
        </Stack>
      </div>
    </DashboardShell>
  );
}
