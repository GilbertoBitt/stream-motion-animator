# Model Integration Guide

## Overview

This guide explains how to integrate new AI animation models into the stream-motion-animator system. The architecture is designed to be extensible, allowing you to add alternative models like AnimateAnyone, SadTalker, or your own custom models.

## Architecture

### Base Model Interface

All animation models must inherit from `BaseAnimationModel`:

```python
from src.animation.base_model import BaseAnimationModel

class YourModel(BaseAnimationModel):
    def load_model(self) -> bool:
        """Load your model"""
        pass
    
    def preprocess_source_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess source image"""
        pass
    
    def animate(self, source_image, driving_landmarks, return_intermediate=False):
        """Generate animated frame"""
        pass
```

### Key Methods

1. **`load_model()`**: Initialize and load model weights
2. **`preprocess_source_image()`**: Prepare source image for model
3. **`animate()`**: Generate animated frame from landmarks
4. **`unload_model()`**: Clean up resources (optional override)

## Step-by-Step Integration

### 1. Create Model Class

Create a new file in `src/animation/`:

```python
# src/animation/your_model.py
import numpy as np
import torch
from typing import Dict, Optional
from .base_model import BaseAnimationModel

class YourModel(BaseAnimationModel):
    """Your custom AI animation model"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        gpu_id: int = 0,
        batch_size: int = 1,
        **kwargs
    ):
        super().__init__(model_path, gpu_id)
        self.batch_size = batch_size
        # Add your custom parameters
        
    def load_model(self) -> bool:
        """Load model from weights file"""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            
            # Initialize your model
            self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
            
            # Load model architecture
            from your_model_package import YourModelClass
            self.model = YourModelClass()
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def preprocess_source_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to model input format"""
        # Resize
        import cv2
        resized = cv2.resize(image, (512, 512))
        
        # Convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def animate(
        self,
        source_image: np.ndarray,
        driving_landmarks: Dict,
        return_intermediate: bool = False
    ) -> np.ndarray:
        """Generate animated frame"""
        if not self.initialized:
            raise RuntimeError("Model not initialized")
        
        # Preprocess source if needed
        if source_image is not None:
            source_tensor = self.preprocess_source_image(source_image)
        else:
            # Use cached source
            source_tensor = self.cached_source
        
        # Extract landmarks
        landmarks = driving_landmarks.get('landmarks')
        if landmarks is None:
            # Return source image if no landmarks
            return source_image
        
        # Convert landmarks to model format
        landmarks_tensor = self._landmarks_to_tensor(landmarks)
        
        # Run inference
        with torch.no_grad():
            output_tensor = self.model(source_tensor, landmarks_tensor)
        
        # Convert output to image
        output_image = self._tensor_to_image(output_tensor)
        
        return output_image
    
    def _landmarks_to_tensor(self, landmarks: np.ndarray) -> torch.Tensor:
        """Convert landmarks to tensor format"""
        # Implement based on your model's requirements
        tensor = torch.from_numpy(landmarks).float().to(self.device)
        return tensor
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert output tensor to BGR image"""
        # Remove batch dimension
        tensor = tensor.squeeze(0)
        
        # Convert to numpy
        image = tensor.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        image = (image * 255).clip(0, 255).astype(np.uint8)
        
        # Convert RGB to BGR
        import cv2
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return bgr
```

### 2. Register Model

Add your model to `src/animation/__init__.py`:

```python
from .base_model import BaseAnimationModel
from .live_portrait import LivePortraitModel
from .your_model import YourModel

__all__ = ['BaseAnimationModel', 'LivePortraitModel', 'YourModel']
```

### 3. Add Configuration

Add model configuration to `config.yaml`:

```yaml
animation:
  model: your_model  # New model type
  model_path: models/your_model.pth
  gpu_id: 0
  batch_size: 4
  
  # Model-specific parameters
  your_model:
    custom_param1: value1
    custom_param2: value2
```

### 4. Update Pipeline

Modify `src/pipeline/animator_pipeline.py` to support your model:

```python
def initialize(self) -> bool:
    # ... existing code ...
    
    # Initialize animator
    model_type = self.config.get('animation.model', 'live_portrait')
    
    if model_type == 'live_portrait':
        self.animator = LivePortraitModel(...)
    elif model_type == 'your_model':
        from src.animation import YourModel
        self.animator = YourModel(
            model_path=self.config.get('animation.model_path'),
            gpu_id=self.config.get('animation.gpu_id', 0),
            batch_size=self.config.get('animation.batch_size', 4),
            # Add model-specific parameters
            custom_param1=self.config.get('animation.your_model.custom_param1'),
            custom_param2=self.config.get('animation.your_model.custom_param2')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # ... rest of initialization ...
```

## Example Integrations

### Example 1: SadTalker (Audio-Driven)

```python
# src/animation/sadtalker.py
class SadTalkerModel(BaseAnimationModel):
    """SadTalker audio-driven animation"""
    
    def __init__(self, model_path, gpu_id, audio_path=None):
        super().__init__(model_path, gpu_id)
        self.audio_path = audio_path
        self.audio_features = None
    
    def load_audio(self, audio_path: str):
        """Load and process audio"""
        # Extract audio features
        pass
    
    def animate(self, source_image, driving_landmarks, return_intermediate=False):
        """Generate frame based on audio features"""
        # Use audio features instead of/in addition to landmarks
        pass
```

### Example 2: AnimateAnyone (Pose-Driven)

```python
# src/animation/animate_anyone.py
class AnimateAnyoneModel(BaseAnimationModel):
    """AnimateAnyone pose-driven animation"""
    
    def __init__(self, model_path, gpu_id, use_dwpose=True):
        super().__init__(model_path, gpu_id)
        self.use_dwpose = use_dwpose
        self.pose_estimator = None
    
    def load_model(self):
        """Load AnimateAnyone and pose estimator"""
        # Load main model
        # Load DWPose or similar
        pass
    
    def animate(self, source_image, driving_landmarks, return_intermediate=False):
        """Generate frame from pose"""
        # Estimate full body pose from landmarks
        # Generate animation
        pass
```

### Example 3: Custom ONNX Model

```python
# src/animation/onnx_model.py
import onnxruntime as ort

class ONNXAnimationModel(BaseAnimationModel):
    """Generic ONNX model wrapper"""
    
    def load_model(self):
        """Load ONNX model"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers
        )
        self.initialized = True
        return True
    
    def animate(self, source_image, driving_landmarks, return_intermediate=False):
        """Run ONNX inference"""
        # Prepare inputs
        inputs = {
            'source': self.preprocess_source_image(source_image),
            'landmarks': self._prepare_landmarks(driving_landmarks)
        }
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Post-process
        return self._postprocess_output(outputs[0])
```

## Testing Your Model

### Unit Tests

Create tests in `tests/test_your_model.py`:

```python
import pytest
import numpy as np
from src.animation import YourModel

def test_model_loading():
    """Test model loads successfully"""
    model = YourModel(model_path='models/your_model.pth', gpu_id=0)
    assert model.load_model()
    assert model.is_initialized()

def test_preprocessing():
    """Test image preprocessing"""
    model = YourModel(model_path='models/your_model.pth', gpu_id=0)
    model.load_model()
    
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed = model.preprocess_source_image(image)
    
    assert processed is not None
    # Add more assertions

def test_animation():
    """Test animation generation"""
    model = YourModel(model_path='models/your_model.pth', gpu_id=0)
    model.load_model()
    
    source = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    landmarks = {
        'landmarks': np.random.rand(468, 2) * 640
    }
    
    output = model.animate(source, landmarks)
    
    assert output is not None
    assert output.shape == source.shape
```

### Integration Testing

Test with the full pipeline:

```bash
# Create test config
cp config.yaml config_test.yaml

# Edit config_test.yaml to use your model
# animation:
#   model: your_model

# Run pipeline
python main.py --image test.jpg --config config_test.yaml
```

## Optimization Tips

### 1. Batch Processing

Process multiple frames in a batch:

```python
def animate_batch(self, source_images, driving_landmarks_list):
    """Process batch of frames"""
    batch_tensor = torch.stack([
        self.preprocess_source_image(img)
        for img in source_images
    ])
    
    with torch.no_grad():
        outputs = self.model(batch_tensor, landmarks_batch)
    
    return [self._tensor_to_image(out) for out in outputs]
```

### 2. Caching

Cache preprocessed source images:

```python
def set_source_image(self, image):
    """Cache source image features"""
    self.cached_source = self.preprocess_source_image(image)
    self.cached_features = self.model.extract_features(self.cached_source)
```

### 3. Async Processing

Use async operations for I/O:

```python
import asyncio

async def load_model_async(self):
    """Asynchronous model loading"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, self._load_model_sync)
```

## Performance Benchmarking

Test your model's performance:

```bash
# Benchmark with your model
python scripts/benchmark.py \
    --image test.jpg \
    --config config_your_model.yaml \
    --duration 60

# Compare with Live Portrait
python scripts/benchmark.py --image test.jpg --duration 60

# Auto-tune batch size
python scripts/benchmark.py \
    --image test.jpg \
    --config config_your_model.yaml \
    --auto-tune
```

## Documentation

Document your model in `docs/models/YOUR_MODEL.md`:

```markdown
# Your Model Integration

## Overview
Brief description of the model and its capabilities.

## Requirements
- Dependencies
- Model weights location
- Minimum GPU requirements

## Installation
Step-by-step setup instructions

## Configuration
Example configuration

## Performance
Expected FPS, latency, VRAM usage

## Limitations
Known issues and constraints

## References
Links to papers, repos, etc.
```

## Publishing

When ready to share:

1. Add model to supported models list in README
2. Create pull request with documentation
3. Provide example weights or download script
4. Include performance benchmarks

## Common Pitfalls

### 1. Memory Leaks

Always clean up:
```python
def unload_model(self):
    if hasattr(self, 'model'):
        del self.model
    torch.cuda.empty_cache()
    super().unload_model()
```

### 2. Device Mismatch

Ensure all tensors are on the same device:
```python
def animate(self, source_image, driving_landmarks, return_intermediate=False):
    source_tensor = source_tensor.to(self.device)
    landmarks_tensor = landmarks_tensor.to(self.device)
```

### 3. Preprocessing Inconsistency

Match training preprocessing:
```python
# If model was trained with specific normalization
def preprocess_source_image(self, image):
    # Use SAME normalization as training
    normalized = (image / 255.0 - 0.5) / 0.5  # Example
```

## Support

For help with model integration:
- Check existing model implementations
- Review base class documentation
- Ask in GitHub Discussions
- Open an issue for bugs

## Contributing

We welcome model integrations! Please:
1. Follow the base model interface
2. Include comprehensive documentation
3. Provide performance benchmarks
4. Add tests
5. Update README with supported models
