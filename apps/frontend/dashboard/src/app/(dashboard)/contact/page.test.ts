import { describe, expect, test } from "bun:test";
import { CONTACT_LINKS } from "./contact-links";

describe("CONTACT_LINKS", () => {
  test("includes package and generated documentation resources", () => {
    expect(CONTACT_LINKS).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "PyPI package",
          href: "https://pypi.org/project/lactationcurve/",
        }),
        expect.objectContaining({
          label: "pdoc API documentation",
          href: "https://bovi-analytics.github.io/bovi/lactationcurve.html",
        }),
      ])
    );
  });
});
