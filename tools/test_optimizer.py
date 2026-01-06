"""
Test script for inference optimizer.

Run this to verify the optimization system is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from inference_optimizer import InferenceOptimizer, CharacterFeatures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_optimizer_initialization():
    """Test optimizer can be initialized."""
    print("\n" + "="*60)
    print("TEST 1: Optimizer Initialization")
    print("="*60)

    try:
        optimizer = InferenceOptimizer(
            device="cuda",
            fp16=True,
            cache_dir="cache/features_test",
            enable_motion_cache=True
        )

        print("✅ Optimizer initialized successfully")
        print(f"   Device: {optimizer.device}")
        print(f"   FP16: {optimizer.fp16}")
        print(f"   Motion cache enabled: {optimizer.motion_cache is not None}")
        return optimizer
    except Exception as e:
        print(f"❌ Failed to initialize optimizer: {e}")
        return None


def test_character_preprocessing(optimizer):
    """Test character pre-processing."""
    print("\n" + "="*60)
    print("TEST 2: Character Pre-Processing")
    print("="*60)

    if optimizer is None:
        print("❌ Skipped (optimizer not initialized)")
        return

    try:
        # Create dummy character image
        dummy_image = np.random.randint(0, 255, (512, 512, 4), dtype=np.uint8)

        start_time = time.time()
        features = optimizer.preprocess_character(
            character_id="test_character",
            character_image=dummy_image,
            model=None  # No model for basic test
        )
        process_time = (time.time() - start_time) * 1000

        print(f"✅ Character pre-processed successfully")
        print(f"   Processing time: {process_time:.2f}ms")
        print(f"   Character ID: {features.character_id}")
        print(f"   Source tensor shape: {features.source_tensor.shape}")
        print(f"   Alpha mask: {'Present' if features.alpha_mask is not None else 'None'}")

        return features
    except Exception as e:
        print(f"❌ Failed to preprocess character: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_feature_caching(optimizer, features):
    """Test feature retrieval from cache."""
    print("\n" + "="*60)
    print("TEST 3: Feature Caching")
    print("="*60)

    if optimizer is None or features is None:
        print("❌ Skipped (prerequisites not met)")
        return

    try:
        # Retrieve from memory cache
        cached = optimizer.get_character_features("test_character")

        if cached is not None:
            print("✅ Features retrieved from memory cache")
            print(f"   Character ID matches: {cached.character_id == features.character_id}")
        else:
            print("❌ Features not found in cache")
    except Exception as e:
        print(f"❌ Failed to retrieve cached features: {e}")


def test_motion_cache(optimizer):
    """Test motion vector caching."""
    print("\n" + "="*60)
    print("TEST 4: Motion Vector Caching")
    print("="*60)

    if optimizer is None or optimizer.motion_cache is None:
        print("❌ Skipped (motion cache not available)")
        return

    try:
        # Create dummy landmarks
        landmarks = np.random.randn(468, 3).astype(np.float32)

        # First access (miss)
        result1 = optimizer.motion_cache.get(landmarks)
        print(f"   First access: {'Hit' if result1 is not None else 'Miss'} ✅")

        # Add to cache
        import torch
        dummy_motion = torch.randn(10, 10)
        optimizer.motion_cache.put(landmarks, dummy_motion)
        print("   Added to cache ✅")

        # Second access (should hit)
        result2 = optimizer.motion_cache.get(landmarks)
        print(f"   Second access: {'Hit ✅' if result2 is not None else 'Miss ❌'}")

        if result2 is not None:
            print("✅ Motion caching working correctly")
        else:
            print("❌ Motion cache not working as expected")

    except Exception as e:
        print(f"❌ Motion cache test failed: {e}")
        import traceback
        traceback.print_exc()


def test_tensor_operations(optimizer):
    """Test tensor conversion operations."""
    print("\n" + "="*60)
    print("TEST 5: Tensor Operations")
    print("="*60)

    if optimizer is None:
        print("❌ Skipped (optimizer not initialized)")
        return

    try:
        # Test image to tensor
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        start_time = time.time()
        tensor = optimizer._image_to_tensor(test_image)
        convert_time = (time.time() - start_time) * 1000

        print(f"✅ Image to tensor conversion: {convert_time:.3f}ms")
        print(f"   Input shape: {test_image.shape}")
        print(f"   Tensor shape: {tensor.shape}")
        print(f"   Device: {tensor.device}")
        print(f"   Dtype: {tensor.dtype}")

        # Test tensor to image
        start_time = time.time()
        image_back = optimizer._tensor_to_image(tensor)
        convert_back_time = (time.time() - start_time) * 1000

        print(f"✅ Tensor to image conversion: {convert_back_time:.3f}ms")
        print(f"   Output shape: {image_back.shape}")

    except Exception as e:
        print(f"❌ Tensor operations test failed: {e}")
        import traceback
        traceback.print_exc()


def test_optimizer_stats(optimizer):
    """Test optimizer statistics."""
    print("\n" + "="*60)
    print("TEST 6: Optimizer Statistics")
    print("="*60)

    if optimizer is None:
        print("❌ Skipped (optimizer not initialized)")
        return

    try:
        stats = optimizer.get_stats()

        print("✅ Statistics retrieved successfully:")
        print(f"   Cached characters: {stats['cached_characters']}")
        print(f"   Motion cache size: {stats['motion_cache_size']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"   Device: {stats['device']}")
        print(f"   FP16: {stats['fp16']}")

    except Exception as e:
        print(f"❌ Statistics test failed: {e}")


def test_cleanup(optimizer):
    """Test cleanup operations."""
    print("\n" + "="*60)
    print("TEST 7: Cleanup")
    print("="*60)

    if optimizer is None:
        print("❌ Skipped (optimizer not initialized)")
        return

    try:
        optimizer.cleanup()
        print("✅ Cleanup completed successfully")

        # Verify cleanup
        print(f"   Character features cleared: {len(optimizer.character_features) == 0}")
        if optimizer.motion_cache:
            print(f"   Motion cache cleared: {len(optimizer.motion_cache.cache) == 0}")

    except Exception as e:
        print(f"❌ Cleanup test failed: {e}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("INFERENCE OPTIMIZER TEST SUITE")
    print("="*60)
    print("Testing the optimization system...")

    # Run tests
    optimizer = test_optimizer_initialization()
    features = test_character_preprocessing(optimizer)
    test_feature_caching(optimizer, features)
    test_motion_cache(optimizer)
    test_tensor_operations(optimizer)
    test_optimizer_stats(optimizer)
    test_cleanup(optimizer)

    # Final summary
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)
    print("All tests completed. Check results above.")
    print("\nIf all tests show ✅, the optimizer is working correctly!")
    print("If any tests show ❌, check the error messages above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

