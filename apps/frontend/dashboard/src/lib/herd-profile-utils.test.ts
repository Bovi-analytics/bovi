import { describe, expect, test } from "bun:test";
import { herdProfileToStats, statsToHerdProfileFields } from "./herd-profile-utils";
import type { HerdProfile } from "@/types/api";

describe("herd profile utilities", () => {
  test("forces data quality to one when saving profile fields", () => {
    const fields = statsToHerdProfileFields([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25]);

    expect(fields.quality_sequence).toBe(1);
  });

  test("hides legacy data quality values when loading a saved profile", () => {
    const profile: HerdProfile = {
      id: 1,
      name: "Legacy profile",
      description: "",
      achieved_21_milk: 0.1,
      achieved_305_milk: 0.2,
      achieved_75_milk: 0.3,
      achieved_milk: 0.4,
      days_dry: 0.5,
      days_in_milk: 0.6,
      days_open: 0.7,
      days_pregnant: 0.8,
      historic_calving_interval: 0.9,
      quality_sequence: 0.25,
      created_at: null,
      updated_at: null,
    };

    expect(herdProfileToStats(profile)[9]).toBe(1);
  });
});
