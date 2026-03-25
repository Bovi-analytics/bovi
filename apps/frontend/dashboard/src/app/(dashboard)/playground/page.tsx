import type { ReactElement } from "react";

export default function PlaygroundPage(): ReactElement {
  return (
    <div>
      <h1 className="text-2xl font-bold text-foreground">Playground</h1>
      <p className="mt-2 text-muted-foreground">
        Upload your own data and fit lactation curve models interactively.
      </p>
    </div>
  );
}
