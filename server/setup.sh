#!/bin/bash

TIMEOUT=10
gunicorn main:app \
          --bind 0.0.0.0:5000 \
          --reload -R \
          --timeout ${TIMEOUT} \
          --access-logfile - \
          --log-file - \
          --env PYTHONUNBUFFERED=1 -k gevent 2>&1