import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteDetector {
  Interpreter? _interpreter;
  bool _isModelLoaded = false;
  final String modelPath;
  final List<String> classLabels;
  static const int inputSize = 640;
  // static const int inputSize = 768;

  TFLiteDetector({
    required this.modelPath,
    required this.classLabels,
  });

  Future<void> loadModel() async {
    try {
      final options = InterpreterOptions();
      options.threads = 4;
      _interpreter = await Interpreter.fromAsset(modelPath, options: options);
      _isModelLoaded = true;
      print('✅ Model loaded successfully from: $modelPath');
    } catch (e) {
      _isModelLoaded = false;
      print('❌ Failed to load model from $modelPath: $e');
      throw Exception('Could not load model: $e');
    }
  }

  Future<DetectionResult> detectFromImagePath(String imagePath) async {
    if (!_isModelLoaded || _interpreter == null) {
      throw Exception('Model not loaded');
    }

    try {
      final bytes = await File(imagePath).readAsBytes();
      final image = img.decodeImage(bytes);
      if (image == null) throw Exception('Could not decode image');

      final input = _preprocessImage(image);
      final int numClasses = classLabels.length;
      final int outputSize = 4 + numClasses; // bbox + class scores
      final output = [List.generate(outputSize, (_) => List.filled(8400, 0.0))];

      _interpreter!.run(input, output);

      return _processYoloOutput(output[0]);
    } catch (e) {
      print('❌ Inference failed: $e');
      throw Exception('Inference failed: $e');
    }
  }

  List<List<List<List<double>>>> _preprocessImage(img.Image image) {
    final resizedImage = img.copyResize(image, width: inputSize, height: inputSize);

    final input = List.generate(1, (_) =>
        List.generate(inputSize, (y) =>
            List.generate(inputSize, (x) {
              final pixel = resizedImage.getPixel(x, y);
              return [
                pixel.r / 255.0,
                pixel.g / 255.0,
                pixel.b / 255.0,
              ];
            })));

    return input;
  }

  DetectionResult _processYoloOutput(List<List<double>> output) {
    double maxConfidence = 0.0;
    int bestClassIndex = -1;
    List<double> bestBbox = [0, 0, 0, 0];

    final int numClasses = classLabels.length;

    for (int i = 0; i < 8400; i++) {
      for (int classIdx = 0; classIdx < numClasses; classIdx++) {
        final double confidence = output[4 + classIdx][i];

        if (confidence > maxConfidence && confidence > 0.3) {
          maxConfidence = confidence;
          bestClassIndex = classIdx;
          bestBbox = [output[0][i], output[1][i], output[2][i], output[3][i]];
        }
      }
    }

    if (bestClassIndex == -1) {
      return DetectionResult(
        detectedName: "No detection found",
        confidence: 0.0,
        isHealthy: true,
        bbox: [0, 0, 0, 0],
      );
    }

    final String detectedName = classLabels[bestClassIndex];
    final bool isHealthy = detectedName.toLowerCase().contains('healthy') || detectedName.toLowerCase().contains('edible');

    return DetectionResult(
      detectedName: detectedName,
      confidence: maxConfidence,
      isHealthy: isHealthy,
      bbox: bestBbox,
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
  final String detectedName;
  final double confidence;
  final bool isHealthy;
  final List<double> bbox;

  DetectionResult({
    required this.detectedName,
    required this.confidence,
    required this.isHealthy,
    required this.bbox,
  });

  @override
  String toString() {
    final confidencePercent = (confidence * 100).toStringAsFixed(1);
    if (isHealthy) {
      return '✅ $detectedName ($confidencePercent% confidence)';
    } else {
      return '⚠️ $detectedName detected ($confidencePercent% confidence)';
    }
  }
}