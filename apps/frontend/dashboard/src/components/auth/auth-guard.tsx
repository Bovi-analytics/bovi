"use client";

import type { ReactNode } from "react";
import { Loader } from "@mantine/core";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useAuth } from "@/lib/auth";

export function AuthGuard({ children }: { readonly children: ReactNode }): ReactNode {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/auth/login");
    }
  }, [isAuthenticated, isLoading, router]);

  if (isLoading) return <Loader />;
  if (!isAuthenticated) return null;
  return children;
}
