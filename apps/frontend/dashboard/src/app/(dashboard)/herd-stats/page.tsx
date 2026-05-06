import type { ReactElement } from "react";
import { Badge, Group, Stack, Text } from "@mantine/core";
import { FlaskConical, Cpu } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { HerdProfileList } from "./components/herd-profile-list";
import { DataSourcePicker } from "./components/data-source-picker";

function SectionHeader({
  icon: Icon,
  title,
  badge,
  description,
}: {
  icon: LucideIcon;
  title: string;
  badge: string;
  description: string;
}): ReactElement {
  return (
    <Stack gap={6}>
      <Group gap="sm" align="center">
        <Icon size={18} className="text-primary" />
        <Text fw={700} size="md">
          {title}
        </Text>
        <Badge size="xs" variant="light" color="violet">
          {badge}
        </Badge>
      </Group>
      <Text size="sm">
        {description}
      </Text>
    </Stack>
  );
}

export default function HerdStatsPage(): ReactElement {
  return (
    <div className="space-y-8 p-6">
      <div>
        <h1 className="text-2xl font-semibold">Herd Data</h1>
        <p className="mt-1 text-sm text-foreground">
          Two types of data live here: cow-level records for curve fitting, and herd-level
          statistics for the autoencoder. Most users only need the first.
        </p>
      </div>

      {/* Section 1 - cow data for classical models */}
      <div className="space-y-4">
        <SectionHeader
          icon={FlaskConical}
          title="Cow data"
          badge="for Classical Models"
          description="Individual cow records - one measurement per test day. Load a preset farm dataset or upload your own CSV. Used by the classical curve models (Wood, Wilmink, etc.)."
        />
        <DataSourcePicker />
      </div>

      <div className="border-t border-border" />

      {/* Section 2 - herd profiles for autoencoder */}
      <div className="space-y-4">
        <SectionHeader
          icon={Cpu}
          title="Herd profiles"
          badge="for Autoencoder"
          description="Ten aggregate statistics that describe your herd as a whole (achieved milk, days in milk, days open, …). The autoencoder uses these as context to improve its predictions. Not needed for classical models."
        />
        <HerdProfileList />
      </div>
    </div>
  );
}
