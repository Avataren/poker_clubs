#!/bin/sh
set -e

# Export all environment variables so supervisor children inherit them
export DATABASE_URL="sqlite:/data/poker.db"
export SERVER_HOST="0.0.0.0"
export SERVER_PORT="3000"

exec supervisord -n -c /etc/supervisor/conf.d/poker.conf
