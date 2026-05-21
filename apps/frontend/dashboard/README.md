# Bovi Dashboard

Next.js dashboard for the Bovi central API.

## Local Development

From this directory:

```bash
bun install
bun dev --port 3000
```

The dashboard reads the API base URL from `NEXT_PUBLIC_API_URL`. When running
outside the root `just dev` command, copy `.env.local.example` to `.env.local`
and set:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Docker

The recommended container workflow is from the repository root, where the root
Compose file runs the API migration job first and then starts the API and
dashboard together:

```bash
docker compose build
docker compose up -d
docker compose logs -f
docker compose down
```

The API container listens on port `8000`; root Compose publishes it to
`http://localhost:8000`. The dashboard listens on port `3000`; root Compose
publishes it to `http://localhost:3000`.

## API URL

The dashboard calls the API from browser-side code. That means
`NEXT_PUBLIC_API_URL` must be a URL that the browser on the host can reach.
For local Docker Compose, the default is:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

The Docker service name `http://api:8000` is only reachable by other containers
inside the Docker network. It is not reachable by the browser on the host, so it
is not a good default for this frontend bundle.

`NEXT_PUBLIC_*` values are embedded by Next.js at build time. If the public API
URL changes, rebuild the dashboard image:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 docker compose build dashboard
docker compose up -d dashboard
```

## Production Build

From this directory:

```bash
bun install --frozen-lockfile
bun run build
bun run start -- -H 0.0.0.0 -p 3000
```
