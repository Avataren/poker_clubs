# =============================================================================
# Stage 1: Build the Rust backend
# =============================================================================
FROM rust:1.88-bookworm AS backend-builder

WORKDIR /build/backend
COPY backend/Cargo.toml backend/Cargo.lock* ./
# Create dummy main to cache dependency builds
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src

COPY backend/src ./src
# Touch main.rs so cargo knows it changed
RUN touch src/main.rs && cargo build --release

# =============================================================================
# Stage 2: Build the Flutter web client
# =============================================================================
FROM ghcr.io/cirruslabs/flutter:stable AS client-builder

WORKDIR /build/client
COPY poker_client/pubspec.yaml poker_client/pubspec.lock ./
RUN flutter pub get

COPY poker_client/ ./

# Build-time config: the Flutter client auto-detects the backend from the
# browser URL at runtime. Nginx proxies API/WS on the same origin.
RUN printf '%s\n' \
  "class AppConfig {" \
  "  static String get serverHost {" \
  "    final uri = Uri.base;" \
  "    if (uri.port == 80 || uri.port == 443 || uri.port == 0) return uri.host;" \
  "    return '\${uri.host}:\${uri.port}';" \
  "  }" \
  "  static String get httpBaseUrl => Uri.base.origin;" \
  "  static String get wsBaseUrl {" \
  "    final uri = Uri.base;" \
  "    final protocol = uri.scheme == 'https' ? 'wss' : 'ws';" \
  "    return '\$protocol://\$serverHost';" \
  "  }" \
  "}" > lib/config.dart && \
  sed -i 's/static const String baseUrl = AppConfig.httpBaseUrl;/static final String baseUrl = AppConfig.httpBaseUrl;/' lib/services/api_service.dart

RUN flutter build web --release

# =============================================================================
# Stage 3: Runtime image with nginx + backend binary
# =============================================================================
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    ca-certificates \
    libssl3 \
    sqlite3 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy backend binary
COPY --from=backend-builder /build/backend/target/release/poker_server /usr/local/bin/poker_server

# Copy Flutter web build
COPY --from=client-builder /build/client/build/web /var/www/html

# Copy nginx config
COPY deploy/nginx.conf /etc/nginx/sites-available/default

# Copy supervisor config and entrypoint
COPY deploy/supervisord.conf /etc/supervisor/conf.d/poker.conf
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create directories for data
RUN mkdir -p /data /models

EXPOSE 80

CMD ["/entrypoint.sh"]
