/// Server configuration for the poker client.
///
/// Change these values to point to a different server.
class AppConfig {
  static const String serverHost = '127.0.0.1:3000';
  static const String httpBaseUrl = 'http://$serverHost';
  static const String wsBaseUrl = 'ws://$serverHost';
}
