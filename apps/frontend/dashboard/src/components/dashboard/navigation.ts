import type { LucideIcon } from "lucide-react";
import {
  ClipboardList,
  Database,
  FlaskConical,
  Home,
  Mail,
  ShieldCheck,
  Trophy,
} from "lucide-react";

export interface NavigationItem {
  readonly label: string;
  readonly href: string;
  readonly icon: LucideIcon;
  readonly adminOnly?: boolean;
}

export const DASHBOARD_NAVIGATION: readonly NavigationItem[] = [
  { label: "Home", href: "/", icon: Home },
  { label: "Data Upload", href: "/data-upload", icon: Database },
  { label: "Herd Profiles", href: "/herd-profiles", icon: ClipboardList },
  { label: "Curves", href: "/curves", icon: FlaskConical },
  { label: "Benchmark", href: "/benchmark", icon: Trophy },
  { label: "Admin", href: "/admin", icon: ShieldCheck, adminOnly: true },
  { label: "Contact", href: "/contact", icon: Mail },
];
