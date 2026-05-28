import { BookOpen, Github, Globe, Handshake, Mail, Package } from "lucide-react";
import type { LucideIcon } from "lucide-react";

export interface ContactLink {
  readonly label: string;
  readonly description: string;
  readonly value: string;
  readonly href: string;
  readonly icon: LucideIcon;
  readonly external?: boolean;
}

export const CONTACT_LINKS: readonly ContactLink[] = [
  {
    label: "GitHub issues",
    description: "Report bugs or request improvements in the public project tracker.",
    value: "github.com/Bovi-analytics/bovi/issues",
    href: "https://github.com/Bovi-analytics/bovi/issues",
    icon: Github,
    external: true,
  },
  {
    label: "Email",
    description: "Contact the maintainers directly for questions that do not fit an issue.",
    value: "mbv32@cornell.edu",
    href: "mailto:mbv32@cornell.edu",
    icon: Mail,
  },
  {
    label: "PyPI package",
    description:
      "Install or inspect the lactationcurve Python package published by the publication workflow.",
    value: "pypi.org/project/lactationcurve",
    href: "https://pypi.org/project/lactationcurve/",
    icon: Package,
    external: true,
  },
  {
    label: "pdoc API documentation",
    description:
      "Read the generated lactationcurve API docs built with pdoc in the documentation workflow and published on GitHub Pages.",
    value: "bovi-analytics.github.io/bovi/lactationcurve.html",
    href: "https://bovi-analytics.github.io/bovi/lactationcurve.html",
    icon: BookOpen,
    external: true,
  },
  {
    label: "Bovi Analytics",
    description: "Visit the broader Bovi Analytics website.",
    value: "bovi-analytics.org",
    href: "https://bovi-analytics.org/",
    icon: Globe,
    external: true,
  },
  {
    label: "ICAR",
    description: "Learn more about the International Committee for Animal Recording.",
    value: "icar.org",
    href: "https://www.icar.org/",
    icon: Handshake,
    external: true,
  },
];
