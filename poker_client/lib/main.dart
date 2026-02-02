import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'services/api_service.dart';
import 'services/websocket_service.dart';
import 'screens/login_screen.dart';
import 'screens/game_screen.dart';
import 'models/club.dart';

void main() {
  runApp(const PokerApp());
}

class PokerApp extends StatelessWidget {
  const PokerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        Provider(create: (_) => ApiService()),
        Provider(create: (_) => WebSocketService()),
      ],
      child: MaterialApp(
        title: 'Texas Hold\'em Poker',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          primarySwatch: Colors.green,
          brightness: Brightness.dark,
          useMaterial3: true,
        ),
        routes: {'/login': (context) => const LoginScreen()},
        onGenerateRoute: (settings) {
          if (settings.name == '/table') {
            final args = settings.arguments as Map<String, dynamic>;
            final table = args['table'] as PokerTable;
            return MaterialPageRoute(
              builder: (context) => GameScreen(
                table: table,
              ),
            );
          }
          return null;
        },
        home: const LoginScreen(),
      ),
    );
  }
}
