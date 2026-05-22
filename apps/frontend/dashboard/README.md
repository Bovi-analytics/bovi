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

Browser-side code calls the local Next.js proxy at `/api/bovi`. The proxy runs
inside the dashboard server and forwards requests to `NEXT_PUBLIC_API_URL` at
runtime. This keeps the Docker image reusable across environments.

For local development outside Docker, the API runs on the host:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For root Docker Compose, the dashboard container uses the internal Docker
network:

```bash
NEXT_PUBLIC_API_URL=http://api:8000
```

Do not read `NEXT_PUBLIC_API_URL` directly from client components. Next.js
embeds public env values into browser bundles at build time; the proxy route is
what makes this value runtime-configurable.

## Production Build

GitHub Actions builds the deployed dashboard image from this package.

From this directory:

```bash
bun install --frozen-lockfile
bun run build
bun run start -- -H 0.0.0.0 -p 3000
```
