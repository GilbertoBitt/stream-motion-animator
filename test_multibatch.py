"""Test the multi-batch character manager."""
import sys
sys.path.insert(0, 'src')

print("="*60)
print("Testing Multi-Batch Character Manager")
print("="*60)
print()

# Test import
try:
    from character_manager_v2 import MultiBatchCharacterManager
    print("✓ MultiBatchCharacterManager imported successfully")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test initialization
try:
    print("\nInitializing character manager...")
    manager = MultiBatchCharacterManager(
        characters_path="assets/characters",
        target_size=(512, 512),
        auto_crop=True,
        preload_all=False,  # Don't preload for testing
        enable_video_processing=True,
        max_frames_per_video=10,
        video_sample_rate=5
    )
    print("✓ Manager initialized")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test character loading
print(f"\nCharacter count: {manager.get_character_count()}")

if manager.get_character_count() > 0:
    print(f"Current character: {manager.get_current_character_name()}")

    # Test loading character
    print("\nLoading current character...")
    character = manager.get_current_character()
    if character:
        print(f"  Name: {character.name}")
        print(f"  Folder: {character.folder_path}")
        print(f"  Image files: {len(character.image_files)}")
        print(f"  Video files: {len(character.video_files)}")
        print(f"  Total references: {character.total_references}")

        # Get primary image
        primary = manager.get_current_character_image()
        if primary is not None:
            print(f"  Primary image shape: {primary.shape}")
        else:
            print("  Primary image: None")

        # Get all references
        refs = manager.get_current_character_references()
        print(f"  Total loaded references: {len(refs)}")
    else:
        print("  No character loaded")

    # Test statistics
    print("\nStatistics:")
    stats = manager.get_character_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test character switching
    if manager.get_character_count() > 1:
        print("\nTesting character switching...")
        print(f"  Current: {manager.get_current_character_name()}")
        manager.next_character()
        print(f"  After next: {manager.get_current_character_name()}")
        manager.prev_character()
        print(f"  After prev: {manager.get_current_character_name()}")
else:
    print("\n⚠ No characters found")
    print("  Create character folders in assets/characters/")
    print("  Or run: python tools/setup_character_structure.py")

print("\n" + "="*60)
print("✅ Multi-Batch Character Manager Test Complete")
print("="*60)

