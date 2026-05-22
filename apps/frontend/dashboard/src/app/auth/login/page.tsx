"use client";

import type { ReactElement } from "react";
import { Alert, Button, Center, Stack, Text } from "@mantine/core";
import { LogIn } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { startLogin, useAuth } from "@/lib/auth";

export default function LoginPage(): ReactElement {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      router.push("/benchmark");
    }
  }, [isAuthenticated, isLoading, router]);

  return (
    <Center mih="100vh" p="md">
      <Stack maw={420} gap="md">
        <Stack gap={4}>
          <h1 className="text-2xl font-semibold">Bovi</h1>
          <Text c="dimmed" size="sm">
            Sign in with your organization account.
          </Text>
        </Stack>
        <Alert color="blue" variant="light">
          Access to Bovi is managed with Microsoft Entra ID.
        </Alert>
        <Button leftSection={<LogIn size={16} />} onClick={() => void startLogin()}>
          Sign in
        </Button>
      </Stack>
    </Center>
  );
}
