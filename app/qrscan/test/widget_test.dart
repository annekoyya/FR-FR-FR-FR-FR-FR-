import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:quishing_scanner/main.dart';

void main() {
  testWidgets('App renders scan page correctly', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const QuishApp());

    // Verify that the scan page title is displayed
    expect(find.text('Scan a QR code'), findsOneWidget);

    // Verify that the instruction text is displayed
    expect(find.text('Point the camera at a QR code'), findsOneWidget);

    // Verify that no result page elements are present initially
    expect(find.text('Likely phishing'), findsNothing);
    expect(find.text('Likely legitimate'), findsNothing);
  });
}