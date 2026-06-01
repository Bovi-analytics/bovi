import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen } from "@testing-library/react";
import React, { useState } from "react";
import { beforeAll, describe, expect, test, vi } from "vitest";
import { DEFAULT_HERD_STATS } from "@/data/herd-stats-metadata";
import { HerdStatsForm } from "./herd-stats-form";

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

function renderEditableHerdStatsForm() {
  function Wrapper() {
    const [values, setValues] = useState<number[]>([...DEFAULT_HERD_STATS]);

    return (
      <MantineProvider>
        <HerdStatsForm values={values} onChange={setValues} showBoth />
      </MantineProvider>
    );
  }

  render(<Wrapper />);
}

describe("HerdStatsForm", () => {
  test("does not clamp raw manual input while a value is still being typed", () => {
    renderEditableHerdStatsForm();

    const rawInput = screen.getByLabelText("305-day milk kg") as HTMLInputElement;

    fireEvent.change(rawInput, { target: { value: "9" } });
    expect(rawInput.value).toBe("9");

    fireEvent.change(rawInput, { target: { value: "9000" } });
    expect(rawInput.value).toBe("9000");
  });

  test("clamps raw manual input on blur", () => {
    renderEditableHerdStatsForm();

    const rawInput = screen.getByLabelText("305-day milk kg") as HTMLInputElement;

    fireEvent.change(rawInput, { target: { value: "9" } });
    fireEvent.blur(rawInput);

    expect(rawInput.value).toBe("2925");
  });
});
