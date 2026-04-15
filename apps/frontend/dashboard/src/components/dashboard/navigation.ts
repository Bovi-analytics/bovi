import type { LucideIcon } from "lucide-react";
import { BarChart3, FlaskConical, Upload } from "lucide-react";

export interface NavigationItem {
  readonly label: string;
  readonly href: string;
  readonly icon: LucideIcon;
}

export const DASHBOARD_NAVIGATION: readonly NavigationItem[] = [
  { label: "Curves", href: "/curves", icon: FlaskConical },
  { label: "Herd Stats", href: "/herd-stats", icon: BarChart3 },
  { label: "Playground", href: "/playground", icon: Upload },
];
