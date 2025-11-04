import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class WeedDetector {
  static const String modelPath = 'assets/models/best_float16.tflite';
  static const int inputSize = 640;

  Interpreter? _interpreter;
  bool _isModelLoaded = false;

  // Your 2 weed classes
  final List<String> weedClasses = ['BroWeed', 'NarWeed'];

  Future<void> loadModel() async {
    final options = InterpreterOptions()..threads = 4;
    _interpreter = await Interpreter.fromAsset(modelPath, options: options);
    _isModelLoaded = true;
  }

  Future<DetectionResult> detect(String imagePath) async {
    if (!_isModelLoaded || _interpreter == null) {
      throw Exception('Weed model not loaded');
    }

    final bytes = await File(imagePath).readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception('Failed to decode image');

    final input = List.generate(
      1,
          (_) => List.generate(
        inputSize,
            (y) => List.generate(
          inputSize,
              (x) {
            final p = img.copyResize(image, width: inputSize, height: inputSize).getPixel(x, y);
            return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
          },
        ),
      ),
    );

    // YOLOv8 output [1, 6, 8400] => 4 bbox + 2 classes
    final output = [List.generate(6, (_) => List.filled(8400, 0.0))];
    _interpreter!.run(input, output);

    return _processOutput(output[0]);
  }

  DetectionResult _processOutput(List<List<double>> out) {
    double bestConf = 0;
    int bestClass = 0;
    for (int i = 0; i < 8400; i++) {
      for (int c = 0; c < weedClasses.length; c++) {
        final conf = out[4 + c][i];
        if (conf > bestConf && conf > 0.3) {
          bestConf = conf;
          bestClass = c;
        }
      }
    }
    return DetectionResult(
      label: weedClasses[bestClass],
      confidence: bestConf,
    );
  }

  bool get isModelLoaded => _isModelLoaded;
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isModelLoaded = false;
  }
}

class DetectionResult {
  final String label;
  final double confidence;
  DetectionResult({required this.label, required this.confidence});

  @override
  String toString() =>
      confidence > 0 ? "⚠️ $label detected (${(confidence * 100).toStringAsFixed(1)}% confidence)"
          : "✅ No weed detected";
}
