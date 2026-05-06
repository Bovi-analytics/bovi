import type { LucideIcon } from "lucide-react";
import { BarChart3, FlaskConical, Home, Trophy } from "lucide-react";

export interface NavigationItem {
  readonly label: string;
  readonly href: string;
  readonly icon: LucideIcon;
}

export const DASHBOARD_NAVIGATION: readonly NavigationItem[] = [
  { label: "Home", href: "/", icon: Home },
  { label: "Herd Stats", href: "/herd-stats", icon: BarChart3 },
  { label: "Curves", href: "/curves", icon: FlaskConical },
  { label: "Benchmark", href: "/benchmark", icon: Trophy },
];
