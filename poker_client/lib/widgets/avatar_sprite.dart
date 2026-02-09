import 'package:flutter/material.dart';

class AvatarSprite extends StatelessWidget {
  static const int rows = 5;
  static const int cols = 5;
  static const int totalAvatars = rows * cols;
  static const String defaultSpriteAsset = 'assets/avatars/portraits_2.png';

  final int avatarIndex;
  final double size;
  final String spriteAsset;

  const AvatarSprite({
    super.key,
    required this.avatarIndex,
    required this.size,
    this.spriteAsset = defaultSpriteAsset,
  });

  @override
  Widget build(BuildContext context) {
    final safeIndex = avatarIndex.clamp(0, totalAvatars - 1);
    final row = safeIndex ~/ cols;
    final col = safeIndex % cols;

    return ClipOval(
      child: SizedBox(
        width: size,
        height: size,
        child: ClipRect(
          child: Transform.translate(
            offset: Offset(-col * size, -row * size),
            child: SizedBox(
              width: size * cols,
              height: size * rows,
              child: Image.asset(
                spriteAsset,
                fit: BoxFit.fill,
                filterQuality: FilterQuality.high,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

