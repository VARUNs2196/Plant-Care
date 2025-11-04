import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class PlantDiseaseDetector {
  static const String modelPath = 'assets/models/best_int8.tflite';
  static const int inputSize = 640;
  // static const int inputSize = 768;
  Interpreter? _interpreter;
  bool _isModelLoaded = false;

  // Your 38 disease classes from YOLOv8 training
  final List<String> diseaseClasses = [
    'Apple - Apple Scab',
    'Apple - Black Rot',
    'Apple - Cedar Apple Rust',
    'Apple - Healthy',
    'Blueberry - Healthy',
    'Cherry - Powdery Mildew',
    'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot',
    'Corn - Common Rust',
    'Corn - Northern Leaf Blight',
    'Corn - Healthy',
    'Grape - Black Rot',
    'Grape - Esca Black Measles',
    'Grape - Leaf Blight',
    'Grape - Healthy',
    'Orange - Citrus Greening',
    'Peach - Bacterial Spot',
    'Peach - Healthy',
    'Pepper Bell - Bacterial Spot',
    'Pepper Bell - Healthy',
    'Potato - Early Blight',
    'Potato - Late Blight',
    'Potato - Healthy',
    'Raspberry - Healthy',
    'Soybean - Healthy',
    'Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch',
    'Strawberry - Healthy',
    'Tomato - Bacterial Spot',
    'Tomato - Early Blight',
    'Tomato - Late Blight',
    'Tomato - Leaf Mold',
    'Tomato - Septoria Leaf Spot',
    'Tomato - Spider Mites',
    'Tomato - Target Spot',
    'Tomato - Yellow Leaf Curl Virus',
    'Tomato - Mosaic Virus',
    'Tomato - Healthy',
  ];

  Future<void> loadModel() async {
    try {
      final options = InterpreterOptions();
      options.threads = 4; // Use 4 threads for better performance
      _interpreter = await Interpreter.fromAsset(modelPath, options: options);
      _isModelLoaded = true;
      print('✅ YOLOv8 plant disease model loaded successfully');
    } catch (e) {
      print('❌ Failed to load YOLOv8 model: $e');
      _isModelLoaded = false;
      throw Exception('Could not load YOLOv8 model: $e');
    }
  }

  Future<DetectionResult> detectFromImagePath(String imagePath) async {
    if (!_isModelLoaded || _interpreter == null) {
      throw Exception('YOLOv8 model not loaded');
    }

    try {
      // Load and decode image
      final bytes = await File(imagePath).readAsBytes();
      final image = img.decodeImage(bytes);
      if (image == null) throw Exception('Could not decode image');

      // Preprocess image for YOLOv8
      final input = _preprocessImage(image);

      // Prepare output tensor for YOLOv8: [1, 42, 8400]
      // 42 = 4 (bbox coordinates) + 38 (class probabilities)
      // 8400 = number of anchor boxes
      final output = [List.generate(42, (_) => List.filled(8400, 0.0))];

      // Run YOLOv8 inference
      _interpreter!.run(input, output);

      // Process YOLOv8 output
      return _processYoloOutput(output[0]);
    } catch (e) {
      print('❌ YOLOv8 inference failed: $e');
      throw Exception('YOLOv8 inference failed: $e');
    }
  }

  // Preprocess image for YOLOv8 (640x640 input)
  List<List<List<List<double>>>> _preprocessImage(img.Image image) {
    // Resize to YOLOv8 input size (640x640)
    final resizedImage = img.copyResize(image, width: inputSize, height: inputSize);

    // Convert to normalized float tensor [1, 640, 640, 3]
    final input = List.generate(1, (_) =>
        List.generate(inputSize, (y) =>
            List.generate(inputSize, (x) {
              final pixel = resizedImage.getPixel(x, y);
              return [
                pixel.r / 255.0, // Red channel normalized
                pixel.g / 255.0, // Green channel normalized
                pixel.b / 255.0, // Blue channel normalized
              ];
            })));

    return input;
  }

  // Process YOLOv8 output to get best detection
  DetectionResult _processYoloOutput(List<List<double>> output) {
    double maxConfidence = 0.0;
    int bestClassIndex = 0;
    List<double> bestBbox = [0, 0, 0, 0];

    // Process all 8400 detection boxes
    for (int i = 0; i < 8400; i++) {
      // Get bounding box coordinates (first 4 values)
      final double x = output[0][i];
      final double y = output[1][i];
      final double w = output[2][i];
      final double h = output[3][i];

      // Get class probabilities (next 38 values)
      for (int classIdx = 0; classIdx < 38; classIdx++) {
        final double confidence = output[4 + classIdx][i];

        // Apply confidence threshold
        if (confidence > maxConfidence && confidence > 0.3) { // 30% confidence threshold
          maxConfidence = confidence;
          bestClassIndex = classIdx;
          bestBbox = [x, y, w, h];
        }
      }
    }

    final String diseaseName = diseaseClasses[bestClassIndex];
    final bool isHealthy = diseaseName.toLowerCase().contains('healthy');

    return DetectionResult(
      diseaseName: diseaseName,
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
  final String diseaseName;
  final double confidence;
  final bool isHealthy;
  final List<double> bbox;

  DetectionResult({
    required this.diseaseName,
    required this.confidence,
    required this.isHealthy,
    required this.bbox,
  });

  @override
  String toString() {
    final confidencePercent = (confidence * 100).toStringAsFixed(1);
    if (isHealthy) {
      return '✅ $diseaseName ($confidencePercent% confidence)';
    } else {
      return '⚠️ $diseaseName detected ($confidencePercent% confidence)';
    }
  }
}
