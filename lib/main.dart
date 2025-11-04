import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'services/tf_lite.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const PlantCareApp());
}

class PlantCareApp extends StatelessWidget {
  const PlantCareApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Plant Care',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        cardTheme: CardThemeData(
          elevation: 8,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
          ),
        ),
      ),
      home: const MainPage(),
    );
  }
}

class MainPage extends StatefulWidget {
  const MainPage({super.key});

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  int _selectedIndex = 0;

  final List<Widget> _pages = const [
    HomeScreen(),
    HistoryScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: AnimatedSwitcher(
        duration: const Duration(milliseconds: 400),
        child: _pages[_selectedIndex],
      ),
      bottomNavigationBar: NavigationBar(
        animationDuration: const Duration(milliseconds: 500),
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(icon: Icon(Icons.home), label: "Home"),
          NavigationDestination(icon: Icon(Icons.history), label: "History"),
          NavigationDestination(icon: Icon(Icons.settings), label: "Settings"),
        ],
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  final List<Map<String, dynamic>> features = const [
    {
      "title": "🌿 Disease Prediction",
      "subtitle": "YOLOv8 AI plant disease detection",
      "icon": Icons.local_hospital,
    },
    {
      "title": "🌱 Weed Detection",
      "subtitle": "Detect weeds in your crops",
      "icon": Icons.agriculture,
    },
    {
      "title": "🍄 Edible or Not?",
      "subtitle": "Check if a mushroom is safe to eat",
      "icon": Icons.fastfood,
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFa8e063), Color(0xFF56ab2f)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "Plant Care 🌱",
                  style: TextStyle(
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 20),
                Expanded(
                  child: ListView.separated(
                    itemCount: features.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 16),
                    itemBuilder: (context, index) {
                      final feature = features[index];
                      return Hero(
                        tag: feature["title"],
                        child: Card(
                          clipBehavior: Clip.antiAlias,
                          child: InkWell(
                            onTap: () {
                              Navigator.push(
                                context,
                                PageRouteBuilder(
                                  transitionDuration: const Duration(milliseconds: 600),
                                  pageBuilder: (_, __, ___) => FeatureScreen(
                                    title: feature["title"],
                                  ),
                                ),
                              );
                            },
                            child: AnimatedContainer(
                              duration: const Duration(milliseconds: 300),
                              padding: const EdgeInsets.all(16),
                              child: Row(
                                children: [
                                  Icon(
                                    feature["icon"],
                                    size: 42,
                                    color: Theme.of(context).colorScheme.primary,
                                  ),
                                  const SizedBox(width: 16),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          feature["title"],
                                          style: const TextStyle(
                                            fontSize: 20,
                                            fontWeight: FontWeight.bold,
                                          ),
                                        ),
                                        const SizedBox(height: 4),
                                        Text(
                                          feature["subtitle"],
                                          style: TextStyle(
                                            fontSize: 14,
                                            color: Colors.grey.shade600,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  const Icon(Icons.arrow_forward_ios, size: 18),
                                ],
                              ),
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class FeatureScreen extends StatefulWidget {
  final String title;
  const FeatureScreen({super.key, required this.title});

  @override
  State<FeatureScreen> createState() => _FeatureScreenState();
}

class _FeatureScreenState extends State<FeatureScreen> {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  String result = "🔎 Loading model...";

  TFLiteDetector? _detector;
  Timer? _inferenceTimer;
  bool _isProcessing = false;

  final List<String> diseaseClasses = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca Black Measles', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Citrus Greening', 'Peach - Bacterial Spot', 'Peach - Healthy',
    'Pepper Bell - Bacterial Spot', 'Pepper Bell - Healthy', 'Potato - Early Blight', 'Potato - Late Blight',
    'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch', 'Strawberry - Healthy', 'Tomato - Bacterial Spot',
    'Tomato - Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot',
    'Tomato - Spider Mites', 'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus',
    'Tomato - Healthy',
  ];

  final List<String> weedClasses = [
    'BroWeed',
    'NarWeed',
  ];

  final List<String> mushroomClasses = [
    'Edible',
    'Poisonous',
  ];

  @override
  void initState() {
    super.initState();
    _initCamera();
    _initializeModel();
  }

  Map<String, dynamic> _getLabelsAndModelPath() {
    if (widget.title.contains("Disease")) {
      return {
        'path': 'assets/models/best_int8.tflite',
        'labels': diseaseClasses,
        'readyMessage': "✅ YOLOv8 Model Ready - Point camera at plant leaves",
        'modelDescription': "🎯 Powered by your YOLOv8 model (86.4% mAP50)",
      };
    } else if (widget.title.contains("Weed")) {
      return {
        'path': 'assets/models/best_float16.tflite',
        'labels': weedClasses,
        'readyMessage': "✅ Weed Detection Model Ready - Point camera at crops",
        'modelDescription': "🎯 Powered by your weed detection model",
      };
    } else if (widget.title.contains("Edible")) {
      return {
        'path': 'assets/models/mushroom.tflite',
        'labels': mushroomClasses,
        'readyMessage': "✅ Mushroom Model Ready - Point camera at mushrooms",
        'modelDescription': "🎯 Powered by your edible mushroom model",
      };
    } else {
      return {
        'path': null,
        'labels': [],
        'readyMessage': "Feature not supported yet.",
        'modelDescription': "",
      };
    }
  }

  Future<void> _initializeModel() async {
    final modelConfig = _getLabelsAndModelPath();
    final String? modelPath = modelConfig['path'];
    final List<String>? labels = modelConfig['labels'];

    if (modelPath == null) {
      setState(() {
        result = modelConfig['readyMessage'];
      });
      return;
    }

    try {
      _detector = TFLiteDetector(
        modelPath: modelPath,
        classLabels: labels!,
      );

      await _detector!.loadModel();
      setState(() {
        result = modelConfig['readyMessage'];
      });
      _startInference();
    } catch (e) {
      setState(() {
        result = "❌ Failed to load model: ${e.toString()}";
      });
    }
  }

  Future<void> _initCamera() async {
    _controller = CameraController(
      cameras.first,
      ResolutionPreset.medium,
    );
    _initializeControllerFuture = _controller!.initialize();
    if (mounted) setState(() {});
  }

  void _startInference() {
    if (_detector != null && _detector!.isModelLoaded) {
      _inferenceTimer = Timer.periodic(const Duration(seconds: 3), (_) {
        if (!_isProcessing && mounted) {
          _runInference();
        }
      });
    }
  }

  // Future<void> _runInference() async {
  //   if (_isProcessing || _controller == null || !_controller!.value.isInitialized || _detector == null) {
  //     return;
  //   }
  //
  //   setState(() {
  //     _isProcessing = true;
  //     result = "🔄 Analyzing...";
  //   });
  //
  //   try {
  //     final XFile imageFile = await _controller!.takePicture();
  //     final DetectionResult detectionResult = await _detector!.detectFromImagePath(imageFile.path);
  //
  //     if (mounted) {
  //       setState(() {
  //         result = detectionResult.toString();
  //         _isProcessing = false;
  //       });
  //     }
  //
  //     await File(imageFile.path).delete();
  //   } catch (e) {
  //     if (mounted) {
  //       setState(() {
  //         result = "❌ Inference failed: ${e.toString()}";
  //         _isProcessing = false;
  //       });
  //     }
  //   }
  // }
  Future<void> _runInference() async {
    if (_isProcessing || _controller == null || !_controller!.value.isInitialized || _detector == null) {
      return;
    }

    setState(() {
      _isProcessing = true;
      result = "🔄 Analyzing...";
    });

    try {
      final XFile imageFile = await _controller!.takePicture();
      final DetectionResult detectionResult = await _detector!.detectFromImagePath(imageFile.path);

      if (mounted) {
        setState(() {
          // Check if the confidence is too low to be considered a valid detection
          if (detectionResult.confidence < 0.5) { // You can adjust this threshold
            result = "No detection found.";
          } else {
            result = detectionResult.toString();
          }
          _isProcessing = false;
        });
      }

      await File(imageFile.path).delete();
    } catch (e) {
      if (mounted) {
        setState(() {
          result = "❌ Inference failed: ${e.toString()}";
          _isProcessing = false;
        });
      }
    }
  }
  @override
  void dispose() {
    _inferenceTimer?.cancel();
    _controller?.dispose();
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Hero(tag: widget.title, child: Text(widget.title)),
        backgroundColor: Theme.of(context).colorScheme.primary,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          Expanded(
            child: _controller == null
                ? const Center(child: CircularProgressIndicator())
                : FutureBuilder(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  return Stack(
                    children: [
                      CameraPreview(_controller!),
                      if (_isProcessing)
                        Container(
                          color: Colors.black.withOpacity(0.4),
                          child: const Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                CircularProgressIndicator(color: Colors.white),
                                SizedBox(height: 16),
                                Text(
                                  "Analyzing...",
                                  style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 18,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                    ],
                  );
                } else {
                  return const Center(child: CircularProgressIndicator());
                }
              },
            ),
          ),
          AnimatedContainer(
            duration: const Duration(milliseconds: 500),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 4,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
            child: Column(
              children: [
                Text(
                  result,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w500,
                  ),
                  textAlign: TextAlign.center,
                ),
                if (widget.title.contains("Disease") || widget.title.contains("Weed") || widget.title.contains("Edible"))
                  const SizedBox(height: 8),
                if (widget.title.contains("Disease") || widget.title.contains("Weed") || widget.title.contains("Edible"))
                  Text(
                    _getLabelsAndModelPath()['modelDescription'],
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey.shade600,
                      fontStyle: FontStyle.italic,
                    ),
                    textAlign: TextAlign.center,
                  ),
              ],
            ),
          )
        ],
      ),
    );
  }
}

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text(
        "📜 History Coming Soon...",
        style: TextStyle(fontSize: 20, fontWeight: FontWeight.w500),
      ),
    );
  }
}

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text(
        "⚙️ Settings Coming Soon...",
        style: TextStyle(fontSize: 20, fontWeight: FontWeight.w500),
      ),
    );
  }
}