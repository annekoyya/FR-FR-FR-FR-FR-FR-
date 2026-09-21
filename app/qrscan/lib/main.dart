import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:mobile_scanner/mobile_scanner.dart';

// CHANGE THIS to your laptop's LAN IP (phone and laptop on the same Wi-Fi).
// Android emulator: http://10.0.2.2:8000
const String kApiBase = 'http://192.168.1.10:8000';

void main() => runApp(const QuishApp());

class QuishApp extends StatelessWidget {
  const QuishApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Quishing Scanner',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const ScanPage(),
    );
  }
}

class ScanPage extends StatefulWidget {
  const ScanPage({super.key});

  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> {
  final MobileScannerController _controller = MobileScannerController();
  bool _busy = false;

  Future<void> _onDetect(BarcodeCapture capture) async {
    if (_busy) return;
    final String? raw = capture.barcodes.isNotEmpty
        ? capture.barcodes.first.rawValue
        : null;
    if (raw == null || raw.isEmpty) return;

    setState(() => _busy = true);
    await _controller.stop();

    try {
      final res = await http
          .post(
            Uri.parse('$kApiBase/scan'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'url': raw}),
          )
          .timeout(const Duration(seconds: 60));

      if (!mounted) return;
      if (res.statusCode != 200) {
        throw Exception('Server returned ${res.statusCode}');
      }
      final data = jsonDecode(res.body) as Map<String, dynamic>;
      await Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => ResultPage(data: data)),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Scan failed: $e')));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
      await _controller.start();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Scan a QR code')),
      body: Stack(
        alignment: Alignment.center,
        children: [
          MobileScanner(controller: _controller, onDetect: _onDetect),
          if (_busy)
            Container(
              color: Colors.black54,
              child: const Center(child: CircularProgressIndicator()),
            ),
          Positioned(
            bottom: 32,
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Text(
                'Point the camera at a QR code',
                style: TextStyle(color: Colors.white),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class ResultPage extends StatelessWidget {
  final Map<String, dynamic> data;
  const ResultPage({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final bool isPhishing = data['verdict'] == 'phishing';
    final double prob = (data['phishing_probability'] as num).toDouble();
    final Color color = isPhishing ? Colors.red.shade700 : Colors.green.shade700;

    return Scaffold(
      appBar: AppBar(title: const Text('Result')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Icon(isPhishing ? Icons.warning_amber_rounded : Icons.verified_user,
                size: 96, color: color),
            const SizedBox(height: 16),
            Text(
              isPhishing ? 'Likely phishing' : 'Likely legitimate',
              textAlign: TextAlign.center,
              style: TextStyle(
                  fontSize: 26, fontWeight: FontWeight.bold, color: color),
            ),
            const SizedBox(height: 24),
            Text('Phishing probability: ${(prob * 100).toStringAsFixed(2)}%',
                style: const TextStyle(fontSize: 16)),
            Text('Threshold: ${data['threshold']}',
                style: const TextStyle(fontSize: 14, color: Colors.grey)),
            const SizedBox(height: 24),
            const Text('Decoded URL',
                style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 4),
            SelectableText('${data['url']}'),
            const Spacer(),
            FilledButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Scan another'),
            ),
          ],
        ),
      ),
    );
  }
}
