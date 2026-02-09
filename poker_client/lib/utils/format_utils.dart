/// Shared formatting utilities used across screens and widgets.
class FormatUtils {
  /// Format a chip count with K/M suffixes for large values.
  static String formatChips(int chips) {
    if (chips >= 1000000) {
      return '${(chips / 1000000).toStringAsFixed(1)}M';
    } else if (chips >= 1000) {
      return '${(chips / 1000).toStringAsFixed(1)}K';
    }
    return chips.toString();
  }

  /// Format a past DateTime as a relative string ("2h ago", "Just now").
  static String formatRelativeTime(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inDays > 0) {
      return '${diff.inDays}d ago';
    } else if (diff.inHours > 0) {
      return '${diff.inHours}h ago';
    } else if (diff.inMinutes > 0) {
      return '${diff.inMinutes}m ago';
    }
    return 'Just now';
  }

  /// Format a future DateTime as a countdown string ("1d 3h", "Soon").
  static String formatCountdown(DateTime dt) {
    final diff = dt.difference(DateTime.now());
    if (diff.inDays > 0) {
      return '${diff.inDays}d ${diff.inHours % 24}h';
    } else if (diff.inHours > 0) {
      return '${diff.inHours}h ${diff.inMinutes % 60}m';
    } else if (diff.inMinutes > 0) {
      return '${diff.inMinutes}m';
    }
    return 'Soon';
  }

  /// Format a DateTime as an absolute "YYYY-MM-DD HH:MM" string.
  static String formatAbsolute(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} '
        '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }
}
