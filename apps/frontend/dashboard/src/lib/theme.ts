import type { MantineColorScheme } from "@mantine/core";
import { Monitor, Moon, Sun } from "lucide-react";
import type { LucideIcon } from "lucide-react";

export const THEME_STORAGE_KEY = "bovi-dashboard-color-scheme";
export const DEFAULT_THEME: MantineColorScheme = "auto";

export interface ThemeOption {
  readonly value: MantineColorScheme;
  readonly label: string;
  readonly icon: LucideIcon;
}

export const THEME_OPTIONS: readonly ThemeOption[] = [
  { value: "auto", label: "System", icon: Monitor },
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
] as const;

export function getThemeOption(value: MantineColorScheme): ThemeOption {
  return THEME_OPTIONS.find((option) => option.value === value) ?? THEME_OPTIONS[0];
}
