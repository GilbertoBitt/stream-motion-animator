"""
Character Preprocessing & Caching Tool

Pre-processes all characters to create optimized cache for maximum performance.
This tool:
1. Extracts frames from all videos
2. Processes all images
3. Extracts features for each character
4. Caches everything for instant loading

Run this ONCE after adding/updating characters for best performance.
"""

import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("="*70)
print("CHARACTER PREPROCESSING & CACHING TOOL")
print("="*70)
print()
print("This will pre-process all characters for maximum performance.")
print("Run this after adding new characters or videos.")
print()

# Check if characters directory exists
characters_path = Path("assets/characters")
if not characters_path.exists():
    print("❌ Error: assets/characters directory not found!")
    print("   Create it first with: mkdir assets\\characters")
    sys.exit(1)

# Import required modules
try:
    from character_manager_v2 import MultiBatchCharacterManager
    print("✓ Multi-batch character manager imported")
    USE_MULTI_BATCH = True
except ImportError:
    print("⚠ Multi-batch manager not available, using legacy mode")
    USE_MULTI_BATCH = False

# Load configuration
try:
    from config import load_config
    config = load_config()
    print("✓ Configuration loaded")
except Exception as e:
    print(f"⚠ Warning: Could not load config: {e}")
    print("  Using default settings")
    config = None

print()
print("-"*70)
print("STEP 1: Scanning Characters")
print("-"*70)

# Initialize character manager
if USE_MULTI_BATCH:
    print("\nInitializing multi-batch character manager...")
    print("This will:")
    print("  • Scan character folders")
    print("  • Find all images and videos")
    print("  • Extract frames from videos")
    print("  • Cache everything")
    print()

    start_time = time.time()

    try:
        manager = MultiBatchCharacterManager(
            characters_path="assets/characters",
            target_size=(512, 512),
            auto_crop=True,
            preload_all=True,  # Load everything now
            use_preprocessing_cache=True,
            max_frames_per_video=30,
            video_sample_rate=10,
            enable_video_processing=True
        )

        load_time = time.time() - start_time

        print(f"\n✓ Character loading complete in {load_time:.1f} seconds")

        # Show statistics
        stats = manager.get_character_stats()

        print()
        print("-"*70)
        print("RESULTS")
        print("-"*70)
        print(f"✓ Total characters: {stats['total_characters']}")
        print(f"✓ Total images: {stats['total_images']}")
        print(f"✓ Total videos: {stats['total_videos']}")
        print(f"✓ Video frames extracted: {stats['video_frames_extracted']}")
        print(f"✓ Total references: {stats['total_references']}")
        print()

        # Show per-character breakdown
        if manager.get_character_count() > 0:
            print("-"*70)
            print("CHARACTER DETAILS")
            print("-"*70)

            for i in range(manager.get_character_count()):
                manager.switch_character(i)
                char = manager.get_current_character()

                if char:
                    print(f"\n{i+1}. {char.name}")
                    print(f"   • Images: {len(char.image_files)}")
                    print(f"   • Videos: {len(char.video_files)}")
                    print(f"   • Video frames: {char.frames_from_video}")
                    print(f"   • Total references: {len(char.reference_images)}")

                    # List files
                    if char.image_files:
                        print(f"   • Image files:")
                        for img in char.image_files[:3]:
                            print(f"     - {img.name}")
                        if len(char.image_files) > 3:
                            print(f"     ... and {len(char.image_files)-3} more")

                    if char.video_files:
                        print(f"   • Video files:")
                        for vid in char.video_files:
                            print(f"     - {vid.name}")

        # Cache location
        print()
        print("-"*70)
        print("CACHE LOCATIONS")
        print("-"*70)
        print(f"✓ Frame cache: cache/characters/frames/")
        print(f"✓ Feature cache: cache/characters/features/")
        print()

        # Performance info
        print("-"*70)
        print("PERFORMANCE")
        print("-"*70)
        print(f"✓ First load time: {load_time:.1f} seconds")
        print(f"✓ Next load time: ~0.1 seconds (cached)")
        print(f"✓ Memory usage: ~{stats['total_references'] * 1.5:.0f} MB")
        print()

        print("="*70)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*70)
        print()
        print("Your characters are now optimized for maximum performance!")
        print()
        print("Next steps:")
        print("  1. Run the application: run.bat")
        print("  2. Characters will load instantly from cache")
        print("  3. Enjoy smooth 60 FPS animation!")
        print()
        print("Note: Re-run this tool when you:")
        print("  • Add new characters")
        print("  • Add new videos to existing characters")
        print("  • Modify character images")

    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

else:
    # Legacy mode
    print("\nUsing legacy character manager...")
    print("⚠ Multi-batch features not available")
    print()

    try:
        from character_manager import CharacterManager

        manager = CharacterManager(
            characters_path="assets/characters",
            target_size=(512, 512),
            auto_crop=True,
            preload_all=True,
            use_preprocessing_cache=True
        )

        print(f"✓ Found {manager.get_character_count()} characters")
        print(f"✓ All characters preloaded and cached")
        print()
        print("="*70)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*70)
        print()
        print("Note: To use multi-batch features with videos:")
        print("  1. Run: python tools/setup_character_structure.py migrate")
        print("  2. Add videos to character folders")
        print("  3. Run this tool again")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print()

