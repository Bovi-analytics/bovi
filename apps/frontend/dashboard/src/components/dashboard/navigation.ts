import type { LucideIcon } from "lucide-react";
import { FlaskConical, Upload, BrainCircuit } from "lucide-react";

export interface NavigationItem {
  readonly label: string;
  readonly href: string;
  readonly icon: LucideIcon;
}

export const DASHBOARD_NAVIGATION: readonly NavigationItem[] = [
  { label: "Models", href: "/models", icon: FlaskConical },
  { label: "Autoencoder", href: "/autoencoder", icon: BrainCircuit },
  { label: "Playground", href: "/playground", icon: Upload },
];
