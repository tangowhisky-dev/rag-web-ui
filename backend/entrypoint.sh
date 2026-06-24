#!/bin/sh

# exit on error
set -e

# Generate a random SECRET_KEY each startup if the placeholder is still set,
# so that JWT tokens are invalidated whenever the container restarts.
if [ -z "$SECRET_KEY" ] || [ "$SECRET_KEY" = "your-secret-key-here" ]; then
  export SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  echo "Generated ephemeral SECRET_KEY for this session"
fi

echo "Waiting for MySQL..."
while ! nc -z db 3306; do
  sleep 1
done
echo "MySQL started"

echo "Running migrations..."
HEADS=$(alembic heads 2>/dev/null | sed 's/ (head)//')
if echo "$HEADS" | grep -q "^$"; then
  echo "No migrations to apply"
elif echo "$HEADS" | grep -q "^$" || echo "$HEADS" | wc -l | grep -q "^1$"; then
  # Single head — normal upgrade (never fall back to stamping)
  if alembic upgrade head 2>/dev/null; then
    echo "Migrations completed successfully"
  else
    echo "Migrations already applied, skipping upgrade"
  fi
else
  # Multiple heads — this is a real migration error; do NOT stamp silently.
  echo "ERROR: Multiple migration heads detected:"
  echo "$HEADS"
  echo "Fix: resolve migration branches or run 'alembic upgrade head' manually."
  exit 1
fi

echo "Starting application..."
if [ "$ENVIRONMENT" = "development" ]; then
  # --reload-dir /app/app: only watch source code changes, NOT /app/uploads or /app/assets.
  # Without this, every file write during ingest (uploads, temp files) triggers a reload
  # that kills the worker mid-flight, leaving all in-progress tasks stuck in "processing".
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app/app --timeout-keep-alive 120
else
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 --timeout-keep-alive 120
fi
