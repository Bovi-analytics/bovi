# syntax=docker/dockerfile:1.7

ARG BUN_VERSION=1.3.4

FROM oven/bun:${BUN_VERSION} AS base
WORKDIR /usr/src/app
ENV NEXT_TELEMETRY_DISABLED=1

FROM base AS deps
COPY package.json bun.lock ./
RUN --mount=type=cache,target=/root/.bun/install/cache \
    bun install --frozen-lockfile --backend copyfile

FROM base AS build
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
COPY --from=deps /usr/src/app/node_modules ./node_modules
COPY . .
RUN bun run build && rm -rf .next/cache

FROM base AS prod-deps
COPY package.json bun.lock ./
RUN --mount=type=cache,target=/root/.bun/install/cache \
    bun install --frozen-lockfile --production --backend copyfile

FROM base AS runtime
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

COPY --from=prod-deps /usr/src/app/node_modules ./node_modules
COPY --from=build /usr/src/app/package.json ./package.json
COPY --from=build /usr/src/app/next.config.mjs ./next.config.mjs
COPY --from=build /usr/src/app/public ./public
COPY --from=build /usr/src/app/.next ./.next

EXPOSE 3000

CMD ["bun", "run", "start", "--", "-H", "0.0.0.0", "-p", "3000"]
