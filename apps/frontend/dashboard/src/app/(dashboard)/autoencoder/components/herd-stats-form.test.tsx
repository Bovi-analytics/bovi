import { cleanup, fireEvent, render, waitFor } from "@testing-library/react";
import { JSDOM } from "jsdom";
import React, { useState } from "react";
import { afterEach, beforeAll, describe, expect, test, vi } from "vitest";
import { DEFAULT_HERD_STATS } from "@/data/herd-stats-metadata";
import { HerdStatsForm } from "./herd-stats-form";

vi.mock("@mantine/core", () => ({
  NumberInput: ({
    "aria-label": ariaLabel,
    label,
    onBlur,
    onChange,
    value,
  }: {
    "aria-label"?: string;
    label?: string;
    onBlur?: () => void;
    onChange?: (value: number) => void;
    value?: number;
  }) => (
    <label>
      {label}
      <input
        aria-label={ariaLabel ?? label}
        onBlur={onBlur}
        onInput={(event) => {
          onChange?.(Number((event.currentTarget as HTMLInputElement).value));
        }}
        value={String(value ?? "")}
      />
    </label>
  ),
  Slider: () => null,
  Tooltip: ({ children }: { children: React.ReactNode }) => children,
}));

const dom = new JSDOM("<!doctype html><html><body></body></html>");

Object.defineProperties(globalThis, {
  window: { value: dom.window },
  document: { value: dom.window.document },
  Element: { value: dom.window.Element },
  HTMLElement: { value: dom.window.HTMLElement },
  HTMLInputElement: { value: dom.window.HTMLInputElement },
  Node: { value: dom.window.Node },
  navigator: { value: dom.window.navigator },
  requestAnimationFrame: { value: (callback: FrameRequestCallback) => setTimeout(callback, 0) },
  cancelAnimationFrame: { value: (handle: number) => clearTimeout(handle) },
});

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
});

afterEach(() => {
  cleanup();
});

function renderEditableHerdStatsForm() {
  function Wrapper() {
    const [values, setValues] = useState<number[]>([...DEFAULT_HERD_STATS]);

    return (
      <HerdStatsForm values={values} onChange={setValues} showBoth />
    );
  }

  return render(<Wrapper />);
}

describe("HerdStatsForm", () => {
  test("does not clamp raw manual input while a value is still being typed", () => {
    const { getByLabelText } = renderEditableHerdStatsForm();

    const rawInput = getByLabelText("305-day milk kg") as HTMLInputElement;

    fireEvent.input(rawInput, { target: { value: "9" } });
    expect(rawInput.value).toBe("9");

    fireEvent.input(rawInput, { target: { value: "9000" } });
    expect(rawInput.value).toBe("9000");
  });

  test("clamps raw manual input on blur", async () => {
    const { getByLabelText } = renderEditableHerdStatsForm();

    const rawInput = getByLabelText("305-day milk kg") as HTMLInputElement;

    fireEvent.input(rawInput, { target: { value: "9" } });
    fireEvent.blur(rawInput);

    await waitFor(() => expect(rawInput.value).toBe("3000"));
  });
});
