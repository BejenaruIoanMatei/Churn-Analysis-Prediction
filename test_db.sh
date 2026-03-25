#!/bin/bash

export $(grep -v '^#' .env | xargs)
echo "Connecting to $POSTGRES_DB as $POSTGRES_USER..."
docker exec postgres_container psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT 'Connection Successful!' as status;"