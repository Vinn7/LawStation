FROM node:22-alpine AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY backend ./backend
COPY mcp_servers ./mcp_servers
COPY alembic.ini ./
COPY alembic ./alembic
RUN pip install --no-cache-dir .
COPY run.py .
COPY data ./data
COPY --from=frontend-build /build/frontend/dist ./frontend/dist
CMD ["python","run.py","--no-build","--host","0.0.0.0","--port","8000"]
