FROM node:20-bullseye-slim AS deps
WORKDIR /app
COPY package*.json ./
RUN npm install --include=dev --no-audit --no-fund

FROM deps AS builder
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /app
ENV NODE_ENV=production
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
EXPOSE 8600
CMD ["node", "dist/index.js"]
