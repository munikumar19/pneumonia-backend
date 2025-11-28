#!/bin/bash
echo "Starting Flask app with Gunicorn..."
exec gunicorn --bind 0.0.0.0:$PORT app:app --workers 1 --threads 4
