import type { LucideIcon } from "lucide-react";
import { ClipboardList, Database, FlaskConical, Home, Settings, Trophy } from "lucide-react";

export interface NavigationItem {
  readonly label: string;
  readonly href: string;
  readonly icon: LucideIcon;
}

export const DASHBOARD_NAVIGATION: readonly NavigationItem[] = [
  { label: "Home", href: "/", icon: Home },
  { label: "Data Upload", href: "/data-upload", icon: Database },
  { label: "Herd Profiles", href: "/herd-profiles", icon: ClipboardList },
  { label: "Curves", href: "/curves", icon: FlaskConical },
  { label: "Benchmark", href: "/benchmark", icon: Trophy },
  { label: "Organization", href: "/organization", icon: Settings },
];
