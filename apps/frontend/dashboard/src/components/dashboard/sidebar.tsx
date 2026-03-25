"use client";

import type { ReactElement } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { DASHBOARD_NAVIGATION } from "./navigation";

export function Sidebar(): ReactElement {
  const pathname = usePathname();

  return (
    <aside className="sticky top-0 hidden h-screen w-52 shrink-0 border-r border-border/40 bg-card/80 p-3 text-sm text-muted-foreground md:flex md:flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 px-3 py-2">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-primary">
            Bovi-Analytics
          </p>
          <h2 className="text-base font-semibold text-foreground">Lactation Curves</h2>
        </div>
      </div>

      {/* Navigation */}
      <nav className="mt-6 flex flex-col gap-1">
        {DASHBOARD_NAVIGATION.map((item) => {
          const isActive = pathname.startsWith(item.href);
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 transition-colors",
                isActive ? "bg-primary/10 text-primary" : "hover:bg-muted/40 hover:text-foreground"
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span className="font-medium tracking-wide">{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
