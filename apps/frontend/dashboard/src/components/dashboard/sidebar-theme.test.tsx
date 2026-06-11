import { cleanup, fireEvent, render } from "@testing-library/react";
import { JSDOM } from "jsdom";
import React from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { Sidebar } from "./sidebar";

let currentColorScheme: "auto" | "light" | "dark" = "auto";
const setColorScheme = vi.fn();

vi.mock("next/navigation", () => ({
  usePathname: () => "/curves",
}));

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

vi.mock("next/image", () => ({
  default: ({
    alt,
    priority: _priority,
  }: React.ImgHTMLAttributes<HTMLImageElement> & { priority?: boolean }) => (
    <span aria-label={alt} role="img" />
  ),
}));

vi.mock("@/lib/auth", () => ({
  useAuth: () => ({
    logout: vi.fn(),
    selectedOrganizationId: 10,
    setSelectedOrganizationId: vi.fn(),
    user: {
      id: 1,
      entra_tenant_id: "tenant",
      entra_oid: "oid",
      account_type: "entra",
      email: "jane.doe@example.test",
      name: "Jane Doe",
      roles: ["Member"],
      is_admin: false,
      organizations: [{ id: 10, name: "North Herd", role: "Owner" }],
    },
  }),
}));

vi.mock("@mantine/core", () => {
  const passthrough =
    (tag: keyof JSX.IntrinsicElements = "div") =>
    ({ children, className }: { children?: React.ReactNode; className?: string }) =>
      React.createElement(tag, { className }, children);

  function Button({ children, onClick }: { children?: React.ReactNode; onClick?: () => void }) {
    return (
      <button onClick={onClick} type="button">
        {children}
      </button>
    );
  }

  function Menu({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }

  Menu.Target = ({ children }: { children: React.ReactNode }) => <>{children}</>;
  Menu.Dropdown = ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
  Menu.Divider = () => <hr />;
  Menu.Label = ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
  Menu.Item = ({
    children,
    disabled,
    onClick,
  }: {
    children: React.ReactNode;
    disabled?: boolean;
    onClick?: () => void;
  }) => (
    <button disabled={disabled} onClick={onClick} type="button">
      {children}
    </button>
  );

  return {
    Avatar: passthrough(),
    Badge: passthrough(),
    Button,
    Group: passthrough(),
    Menu,
    Select: () => null,
    Stack: passthrough(),
    Text: passthrough("span"),
    UnstyledButton: passthrough("button"),
    useComputedColorScheme: () => (currentColorScheme === "dark" ? "dark" : "light"),
    useMantineColorScheme: () => ({
      colorScheme: currentColorScheme,
      setColorScheme,
    }),
  };
});

const dom = new JSDOM("<!doctype html><html><body></body></html>");

Object.defineProperties(globalThis, {
  window: { configurable: true, value: dom.window },
  document: { configurable: true, value: dom.window.document },
  Element: { configurable: true, value: dom.window.Element },
  HTMLElement: { configurable: true, value: dom.window.HTMLElement },
  HTMLButtonElement: { configurable: true, value: dom.window.HTMLButtonElement },
  HTMLImageElement: { configurable: true, value: dom.window.HTMLImageElement },
  Node: { configurable: true, value: dom.window.Node },
  navigator: { configurable: true, value: dom.window.navigator },
});

afterEach(() => {
  cleanup();
  setColorScheme.mockReset();
  currentColorScheme = "auto";
});

describe("Sidebar theme menu", () => {
  test("renders theme choices in the user menu", () => {
    const { getByText } = render(<Sidebar />);

    expect(getByText("System")).not.toBeNull();
    expect(getByText("Light")).not.toBeNull();
    expect(getByText("Dark")).not.toBeNull();
    expect(getByText("Active: Light")).not.toBeNull();
  });

  test("sets the selected Mantine color scheme", () => {
    const { getByText } = render(<Sidebar />);

    fireEvent.click(getByText("Dark"));

    expect(setColorScheme).toHaveBeenCalledWith("dark");
  });
});
