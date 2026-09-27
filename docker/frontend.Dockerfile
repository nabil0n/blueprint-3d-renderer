# Development image for the Vite + React frontend. Build context: repository root
# (the frontend imports the shared sample plan from tests/fixtures).
FROM node:22-alpine

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
COPY tests/fixtures /app/tests/fixtures

EXPOSE 5173
CMD ["npm", "run", "dev"]
