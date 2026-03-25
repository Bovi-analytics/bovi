import type { LucideIcon } from "lucide-react";
import { FlaskConical, Upload } from "lucide-react";

export interface NavigationItem {
  readonly label: string;
  readonly href: string;
  readonly icon: LucideIcon;
}

export const DASHBOARD_NAVIGATION: readonly NavigationItem[] = [
  { label: "Models", href: "/models", icon: FlaskConical },
  { label: "Playground", href: "/playground", icon: Upload },
];
