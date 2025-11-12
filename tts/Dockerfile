FROM node:20-bullseye-slim AS deps
WORKDIR /app
COPY package*.json ./
RUN npm install --include=dev --no-audit --no-fund

FROM deps AS builder
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:20-bullseye-slim
WORKDIR /app
ENV NODE_ENV=production
COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
EXPOSE 8600
CMD ["node", "dist/index.js"]
