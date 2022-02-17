#!/bin/bash

TIMEOUT=30
gunicorn main:app -w 2 \
          --bind 0.0.0.0:5000 \
          --reload -R \
          --timeout ${TIMEOUT} \
          --access-logfile - \
          --log-file - \
          --worker-connections 100 \
          --env PYTHONUNBUFFERED=1 -k gevent 2>&1