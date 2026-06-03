import { cleanup, render } from "@testing-library/react";
import { JSDOM } from "jsdom";
import React from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { AuthGuard } from "./auth-guard";
import type { AuthUser } from "@/lib/auth/types";

const mockPush = vi.fn();
let mockAuthState: {
  isAuthenticated: boolean;
  isLoading: boolean;
  setSelectedOrganizationId: (organizationId: number | "all" | null) => void;
  user: AuthUser | null;
};

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
}));

vi.mock("@mantine/core", () => ({
  Button: ({
    children,
    loading,
    onClick,
  }: {
    children: React.ReactNode;
    loading?: boolean;
    onClick?: () => void;
  }) => (
    <button aria-busy={loading ? "true" : "false"} onClick={onClick} type="button">
      {children}
    </button>
  ),
  Stack: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TextInput: ({
    label,
    onChange,
    placeholder,
    value,
  }: {
    label?: string;
    onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void;
    placeholder?: string;
    value?: string;
  }) => (
    <label>
      {label}
      <input onChange={onChange} placeholder={placeholder} value={value} />
    </label>
  ),
  Title: ({ children }: { children: React.ReactNode }) => <h1>{children}</h1>,
}));

vi.mock("@/components/dashboard/centered-loader", () => ({
  CenteredLoader: ({ label }: { label: string }) => <div>{label}</div>,
}));

vi.mock("@/lib/api-client", () => ({
  createOrganization: vi.fn(),
}));

vi.mock("@/lib/auth", () => ({
  useAuth: () => mockAuthState,
}));

const dom = new JSDOM("<!doctype html><html><body></body></html>");

Object.defineProperties(globalThis, {
  window: { configurable: true, value: dom.window, writable: true },
  document: { configurable: true, value: dom.window.document, writable: true },
  Element: { configurable: true, value: dom.window.Element, writable: true },
  HTMLElement: { configurable: true, value: dom.window.HTMLElement, writable: true },
  HTMLInputElement: { configurable: true, value: dom.window.HTMLInputElement, writable: true },
  Node: { configurable: true, value: dom.window.Node, writable: true },
  navigator: { configurable: true, value: dom.window.navigator, writable: true },
});

const USER_WITHOUT_ORG: AuthUser = {
  id: 2,
  entra_tenant_id: "tenant",
  entra_oid: "oid",
  account_type: "personal",
  email: "dekokdouwe@gmail.com",
  name: "Douwe de Kok",
  roles: ["User"],
  is_admin: false,
  organizations: [],
};

const USER_WITH_ORG: AuthUser = {
  ...USER_WITHOUT_ORG,
  organizations: [{ id: 42, name: "Bovi Dairy", role: "Owner" }],
};

afterEach(() => {
  cleanup();
  mockPush.mockReset();
});

describe("AuthGuard", () => {
  test("shows the create organization card for authenticated users without organizations", () => {
    mockAuthState = {
      isAuthenticated: true,
      isLoading: false,
      setSelectedOrganizationId: vi.fn(),
      user: USER_WITHOUT_ORG,
    };

    const { container, queryByText } = render(
      <AuthGuard>
        <div>Protected dashboard</div>
      </AuthGuard>
    );

    expect(queryByText("Create your Bovi organization")).not.toBeNull();
    expect(container.querySelector("input")?.getAttribute("placeholder")).toBe("Your Organization");
    expect(queryByText("Create organization")).not.toBeNull();
    expect(queryByText("Protected dashboard")).toBeNull();
  });

  test("renders protected content when the authenticated user belongs to an organization", () => {
    mockAuthState = {
      isAuthenticated: true,
      isLoading: false,
      setSelectedOrganizationId: vi.fn(),
      user: USER_WITH_ORG,
    };

    const { queryByText } = render(
      <AuthGuard>
        <div>Protected dashboard</div>
      </AuthGuard>
    );

    expect(queryByText("Protected dashboard")).not.toBeNull();
    expect(queryByText("Create your Bovi organization")).toBeNull();
  });
});
