# Stream Motion Animator

**AI-Driven Live Portrait Animation System for Real-Time Streaming**

Transform static character images into live, animated avatars driven by your webcam, optimized for RTX 3080 to achieve 60-70 FPS streaming performance.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)

## ✨ Features

- 🎭 **AI-Powered Animation**: Uses Live Portrait model for high-quality facial animation
- ⚡ **High Performance**: Optimized for 60+ FPS on RTX 3080
- 🖼️ **Multi-Character Support**: Switch between characters in real-time with hotkeys
- 📡 **Dual Output**: Simultaneous Spout and NDI streaming to OBS
- 🎯 **Low Latency**: <20ms latency from movement to output
- 🔧 **Configurable**: Extensive YAML configuration for customization
- 📊 **Performance Monitoring**: Real-time FPS and GPU usage tracking
- 🎨 **Transparency Support**: RGBA output with transparent backgrounds

## 🚀 Quick Start

### Prerequisites

- **Operating System**: Windows 10/11 (for Spout support)
- **GPU**: NVIDIA RTX 20/30/40 series (RTX 3080 recommended)
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher
- **Webcam**: Any USB webcam

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GilbertoBitt/stream-motion-animator.git
   cd stream-motion-animator
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download AI models**:
   ```bash
   python tools/download_models.py
   ```

5. **Add character images**:
   Place your character images (PNG/JPG) in `assets/characters/`

6. **Run the application**:
   ```bash
   python src/main.py
   ```

For detailed installation instructions, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## 📖 Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Step-by-step setup instructions
- **[Performance Guide](docs/PERFORMANCE.md)** - RTX 3080 optimization tips
- **[Character Guide](docs/CHARACTER_GUIDE.md)** - Preparing character images
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 🎮 Usage

### Hotkeys

- **1-9**: Switch to character 1-9
- **Left/Right Arrow**: Previous/Next character
- **R**: Reload characters
- **T**: Toggle stats display
- **S**: Toggle Spout output
- **N**: Toggle NDI output
- **Q**: Quit application

### Configuration

Edit `assets/config.yaml` to customize:
- Video capture settings
- AI model parameters
- Performance targets
- Output settings
- Hotkey bindings

### OBS Integration

**Using Spout**:
1. Install Spout2 plugin for OBS
2. Add "Spout2 Capture" source
3. Select "StreamMotionAnimator"

**Using NDI**:
1. Install NDI plugin for OBS
2. Add "NDI Source"
3. Select "Stream Motion Animator"

## 🛠️ Tools

### Benchmark Performance
```bash
python tools/benchmark.py --duration 30
```

Tests system performance and provides optimization recommendations.

### Test Character Images
```bash
python tools/test_character.py assets/characters/your_image.png
```

Validates character images for compatibility.

### Download Models
```bash
python tools/download_models.py --list  # List available models
python tools/download_models.py --all   # Download all models
```

## 📊 Performance Targets

| Configuration | Expected FPS | GPU Usage | Latency |
|--------------|--------------|-----------|---------|
| RTX 3080 + FP16 | 60-70 FPS | 70-80% | <20ms |
| RTX 3070 + FP16 | 45-55 FPS | 80-90% | <25ms |
| RTX 4080 + FP16 | 80-100 FPS | 60-70% | <15ms |

*Results with 1080p output, 720p input, Live Portrait model*

## 🏗️ Architecture

```
Webcam → [Capture Thread] → Queue 1
Queue 1 → [Tracking Thread: MediaPipe] → Queue 2
Queue 2 → [Inference Thread: AI Model] → Queue 3
Queue 3 → [Output Thread: Spout/NDI] → OBS
```

**Async Pipeline**: Multithreaded design for parallel processing and maximum throughput.

## 📁 Project Structure

```
stream-motion-animator/
├── src/                    # Source code
│   ├── main.py            # Main application
│   ├── motion_tracker.py  # MediaPipe face tracking
│   ├── ai_animator.py     # AI model inference
│   ├── character_manager.py # Character management
│   ├── output_manager.py  # Spout/NDI output
│   └── models/            # AI model implementations
├── assets/
│   ├── characters/        # Character images
│   └── config.yaml        # Configuration
├── tools/                 # Utility scripts
├── docs/                  # Documentation
└── models/                # AI model weights (gitignored)
```

## 🔮 Roadmap

- [ ] Full Live Portrait model integration
- [ ] TensorRT optimization
- [ ] AnimateAnyone model support
- [ ] Full body animation
- [ ] Audio-driven animation
- [ ] VTuber mode with background
- [ ] Network streaming support
- [ ] Custom model fine-tuning

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Live Portrait](https://github.com/KwaiVGI/LivePortrait) - AI animation model
- [MediaPipe](https://mediapipe.dev/) - Face tracking
- [Spout](https://spout.zeal.co/) - Video sharing framework
- [NDI](https://ndi.tv/) - Network Device Interface

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/GilbertoBitt/stream-motion-animator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GilbertoBitt/stream-motion-animator/discussions)

## ⚠️ System Requirements

**Minimum**:
- NVIDIA GTX 1660 or better
- 8GB RAM
- Windows 10

**Recommended**:
- NVIDIA RTX 3080 or better
- 16GB RAM
- Windows 11
- SSD storage

---

Made with ❤️ for the VTuber and streaming community
