import type { ReactElement } from "react";
import { HerdProfileList } from "./components/herd-profile-list";

export default function HerdStatsPage(): ReactElement {
  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold">Herd Stats</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Manage saved herd profiles. Select a profile on the Curves page to use its statistics as
          a starting point for autoencoder predictions.
        </p>
      </div>
      <HerdProfileList />
    </div>
  );
}
