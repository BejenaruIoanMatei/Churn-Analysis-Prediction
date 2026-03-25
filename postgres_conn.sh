#!/bin/bash

export $(grep -v '^#' .env | xargs)
echo "Connecting to $POSTGRES_DB as $POSTGRES_USER..."
docker-compose exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB