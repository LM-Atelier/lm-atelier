FROM node:22-bookworm-slim AS web
WORKDIR /source
COPY package.json package-lock.json ./
COPY apps/web/package.json apps/web/package.json
RUN npm ci
COPY apps/web apps/web
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LOCAL_LM_DATA_DIR=/data LOCAL_LM_WEB_DIST_DIR=/app/apps/web/dist
WORKDIR /app
COPY services/api services/api
RUN pip install --no-cache-dir ./services/api
COPY --from=web /source/apps/web/dist apps/web/dist
EXPOSE 12340
VOLUME ["/data"]
CMD ["lm-atelier"]
