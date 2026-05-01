/**
 * Sample sparse test-day points from dense daily milk recordings.
 *
 * Mimics real-world MPR (milk production registration) sampling where
 * cows are measured approximately every 4-5 weeks. This allows classical
 * lactation curve models to be fitted on the same cow data that the
 * autoencoder uses.
 */
export function sampleTestDays(
  milk: readonly (number | null)[],
  intervalDays = 35
): { dim: number[]; milkrecordings: number[] } {
  const dim: number[] = [];
  const milkrecordings: number[] = [];
  const searchRadius = 3;

  for (let target = 0; target < milk.length; target += intervalDays) {
    // Search ±searchRadius days around the target for a non-null value
    let bestOffset = 0;
    let found = false;

    for (let offset = 0; offset <= searchRadius; offset++) {
      if (target + offset < milk.length && milk[target + offset] !== null) {
        bestOffset = offset;
        found = true;
        break;
      }
      if (offset > 0 && target - offset >= 0 && milk[target - offset] !== null) {
        bestOffset = -offset;
        found = true;
        break;
      }
    }

    if (found) {
      const idx = target + bestOffset;
      dim.push(idx + 1); // 1-indexed (DIM starts at day 1)
      milkrecordings.push(milk[idx] as number);
    }
  }

  return { dim, milkrecordings };
}
