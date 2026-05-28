import type { ReactElement, ReactNode } from "react";
import { AuthGuard } from "@/components/auth/auth-guard";
import { DashboardShell } from "@/components/dashboard/dashboard-shell";

interface DashboardLayoutProps {
  readonly children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps): ReactElement {
  return (
    <AuthGuard>
      <DashboardShell>{children}</DashboardShell>
    </AuthGuard>
  );
}
