#!/usr/bin/env bash
set -e

echo "Stopping containers..."
docker compose -f docker-compose.dev.yml down

echo "Removing volumes..."
docker volume rm -f \
  rag-web-ui_mysql_data \
  rag-web-ui_chroma_data \
  rag-web-ui_minio_data

echo "Starting fresh..."
docker compose -f docker-compose.dev.yml up -d

echo "Waiting for backend..."
until curl -s http://localhost:8000/api/health > /dev/null 2>&1; do
  sleep 1
done

echo "Done. System is reset and ready."
