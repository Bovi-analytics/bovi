import type { ReactElement } from "react";
import {
  Badge,
  Button,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import { ExternalLink, Github, Globe, Handshake, Mail } from "lucide-react";
import type { LucideIcon } from "lucide-react";

interface ContactLink {
  readonly label: string;
  readonly description: string;
  readonly href: string;
  readonly icon: LucideIcon;
  readonly external?: boolean;
}

const CONTACT_LINKS: readonly ContactLink[] = [
  {
    label: "GitHub issues",
    description: "Report bugs or request improvements in the public project tracker.",
    href: "https://github.com/Bovi-analytics/bovi/issues",
    icon: Github,
    external: true,
  },
  {
    label: "Email",
    description: "Contact the maintainers directly for questions that do not fit an issue.",
    href: "mailto:mbv32@cornell.edu",
    icon: Mail,
  },
  {
    label: "Bovi Analytics",
    description: "Visit the broader Bovi Analytics website.",
    href: "https://bovi-analytics.org/",
    icon: Globe,
    external: true,
  },
  {
    label: "ICAR",
    description: "Learn more about the International Committee for Animal Recording.",
    href: "https://www.icar.org/",
    icon: Handshake,
    external: true,
  },
];

export default function ContactPage(): ReactElement {
  return (
    <div className="space-y-8 p-6">
      <Group justify="space-between" align="flex-start" gap="md">
        <div>
          <Group gap="xs" mb="xs">
            <Badge variant="light" color="blue">
              MIT licensed
            </Badge>
            <Badge variant="light" color="cyan">
              Community feedback
            </Badge>
          </Group>
          <Title order={1}>Contact</Title>
          <Text mt="xs" maw={760} size="sm">
            We encourage everyone to use, test, and contribute to this platform, which is available
            under the MIT license. We welcome feedback and suggestions for improvement, and we are
            committed to maintaining and updating the platform to ensure it remains a valuable
            resource for the dairy industry and research community.
          </Text>
        </div>
      </Group>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
        {CONTACT_LINKS.map((item) => {
          const Icon = item.icon;

          return (
            <Paper key={item.href} withBorder radius="md" p="lg">
              <Stack gap="md" h="100%">
                <Group gap="sm" align="flex-start" wrap="nowrap">
                  <ThemeIcon variant="light" size="lg" radius="md">
                    <Icon size={18} />
                  </ThemeIcon>
                  <div>
                    <Title order={3}>{item.label}</Title>
                    <Text size="sm" c="dimmed" mt={4}>
                      {item.description}
                    </Text>
                  </div>
                </Group>

                <Button
                  component="a"
                  href={item.href}
                  target={item.external ? "_blank" : undefined}
                  rel={item.external ? "noreferrer" : undefined}
                  variant="light"
                  mt="auto"
                  rightSection={item.external ? <ExternalLink size={14} /> : undefined}
                >
                  Open {item.label}
                </Button>
              </Stack>
            </Paper>
          );
        })}
      </SimpleGrid>

      <Paper withBorder radius="md" p="lg">
        <Group gap="sm" align="flex-start" wrap="nowrap">
          <ThemeIcon variant="light" color="cyan" size="lg" radius="md">
            <Handshake size={18} />
          </ThemeIcon>
          <Stack gap="xs">
            <Title order={2}>Acknowledgement</Title>
            <Text size="sm">
              Bovi acknowledges the support of ICAR, whose contribution helped make this lactation
              curve platform available to the dairy industry and research community.
            </Text>
          </Stack>
        </Group>
      </Paper>
    </div>
  );
}
