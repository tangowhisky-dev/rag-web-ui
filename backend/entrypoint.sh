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
if alembic upgrade head; then
  echo "Migrations completed successfully"
else
  echo "Migration failed"
  exit 1
fi

echo "Starting application..."
if [ "$ENVIRONMENT" = "development" ]; then
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
else
  uvicorn app.main:app --host 0.0.0.0 --port 8000
fi
