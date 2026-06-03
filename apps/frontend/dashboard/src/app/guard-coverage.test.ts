import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const APP_DIR = join(import.meta.dir);
const DASHBOARD_DIR = join(APP_DIR, "(dashboard)");

const PROTECTED_ROUTES = [
  "admin",
  "autoencoder",
  "benchmark",
  "contact",
  "curves",
  "data-upload",
  "herd-profiles",
  "herd-stats",
  "models",
  "organization",
];

function readAppFile(...parts: string[]): string {
  return readFileSync(join(APP_DIR, ...parts), "utf8");
}

describe("dashboard organization guard coverage", () => {
  test("wraps the dashboard route group in AuthGuard", () => {
    const layout = readFileSync(join(DASHBOARD_DIR, "layout.tsx"), "utf8");

    expect(layout).toContain('from "@/components/auth/auth-guard"');
    expect(layout).toContain("<AuthGuard>");
    expect(layout).toContain("</AuthGuard>");
  });

  test("keeps all protected dashboard routes inside the guarded route group", () => {
    for (const route of PROTECTED_ROUTES) {
      expect(existsSync(join(DASHBOARD_DIR, route, "page.tsx"))).toBe(true);
    }
  });

  test("guards the authenticated root landing page before rendering dashboard content", () => {
    const rootPage = readAppFile("page.tsx");

    expect(rootPage).toContain('from "@/components/auth/auth-guard"');
    expect(rootPage).toContain("if (isAuthenticated) return <AuthGuard>{content}</AuthGuard>;");
  });
});
