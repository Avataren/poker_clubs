import 'package:flutter/material.dart';

/// A reusable confirmation dialog that returns true/false.
class ConfirmationDialog extends StatelessWidget {
  final String title;
  final String content;
  final String confirmLabel;
  final Color? confirmColor;
  final String cancelLabel;

  const ConfirmationDialog({
    super.key,
    required this.title,
    required this.content,
    this.confirmLabel = 'Confirm',
    this.confirmColor,
    this.cancelLabel = 'Cancel',
  });

  /// Show a confirmation dialog and return true if confirmed.
  static Future<bool> show(
    BuildContext context, {
    required String title,
    required String content,
    String confirmLabel = 'Confirm',
    Color? confirmColor,
    String cancelLabel = 'Cancel',
  }) async {
    final result = await showDialog<bool>(
      context: context,
      builder: (_) => ConfirmationDialog(
        title: title,
        content: content,
        confirmLabel: confirmLabel,
        confirmColor: confirmColor,
        cancelLabel: cancelLabel,
      ),
    );
    return result ?? false;
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Text(title),
      content: Text(content),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context, false),
          child: Text(cancelLabel),
        ),
        ElevatedButton(
          onPressed: () => Navigator.pop(context, true),
          style: confirmColor != null
              ? ElevatedButton.styleFrom(
                  backgroundColor: confirmColor,
                  foregroundColor: Colors.white,
                )
              : null,
          child: Text(confirmLabel),
        ),
      ],
    );
  }
}

/// A reusable input dialog with a text field that returns the entered value.
class InputDialog extends StatelessWidget {
  final String title;
  final String prompt;
  final TextEditingController controller;
  final String? hintText;
  final String? prefixText;
  final String confirmLabel;
  final TextInputType keyboardType;

  const InputDialog({
    super.key,
    required this.title,
    required this.prompt,
    required this.controller,
    this.hintText,
    this.prefixText,
    this.confirmLabel = 'OK',
    this.keyboardType = TextInputType.text,
  });

  /// Show an input dialog and return the text value, or null if cancelled.
  static Future<String?> show(
    BuildContext context, {
    required String title,
    required String prompt,
    required TextEditingController controller,
    String? hintText,
    String? prefixText,
    String confirmLabel = 'OK',
    TextInputType keyboardType = TextInputType.text,
  }) {
    return showDialog<String>(
      context: context,
      builder: (_) => InputDialog(
        title: title,
        prompt: prompt,
        controller: controller,
        hintText: hintText,
        prefixText: prefixText,
        confirmLabel: confirmLabel,
        keyboardType: keyboardType,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Text(title),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(prompt),
          const SizedBox(height: 8),
          TextField(
            controller: controller,
            keyboardType: keyboardType,
            decoration: InputDecoration(
              hintText: hintText,
              prefixText: prefixText,
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        ElevatedButton(
          onPressed: () => Navigator.pop(context, controller.text),
          child: Text(confirmLabel),
        ),
      ],
    );
  }
}
