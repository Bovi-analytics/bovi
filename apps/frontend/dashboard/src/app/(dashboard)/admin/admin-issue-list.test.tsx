import { cleanup, render } from "@testing-library/react";
import { JSDOM } from "jsdom";
import React from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { AdminIssueList } from "./admin-issue-list";

vi.mock("@mantine/core", () => ({
  Alert: ({ children, title }: { children: React.ReactNode; title: string }) => (
    <section aria-label={title}>{children}</section>
  ),
  Code: ({ children }: { children: React.ReactNode }) => <code>{children}</code>,
  Group: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Stack: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Text: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
}));

const dom = new JSDOM("<!doctype html><html><body></body></html>");

Object.defineProperties(globalThis, {
  window: { configurable: true, value: dom.window, writable: true },
  document: { configurable: true, value: dom.window.document, writable: true },
  Element: { configurable: true, value: dom.window.Element, writable: true },
  HTMLElement: { configurable: true, value: dom.window.HTMLElement, writable: true },
  Node: { configurable: true, value: dom.window.Node, writable: true },
  navigator: { configurable: true, value: dom.window.navigator, writable: true },
});

afterEach(cleanup);

describe("AdminIssueList", () => {
  test("shows the explanation, affected count, and sample identifiers", () => {
    const { queryByText } = render(
      <AdminIssueList
        issues={[
          {
            severity: "warning",
            title: "Invalid result rows excluded",
            message: "Two cows were excluded from the benchmark statistics.",
            affected_count: 2,
            sample_ids: ["cow-17", "cow-23"],
          },
        ]}
      />
    );

    expect(queryByText("Two cows were excluded from the benchmark statistics.")).not.toBeNull();
    expect(queryByText("Affected: 2")).not.toBeNull();
    expect(queryByText("cow-17, cow-23")).not.toBeNull();
  });

  test("renders nothing when an item has no issues", () => {
    const { container } = render(<AdminIssueList issues={[]} />);

    expect(container.childElementCount).toBe(0);
  });
});
