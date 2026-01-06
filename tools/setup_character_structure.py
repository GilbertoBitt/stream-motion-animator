"""
Setup tool for multi-batch character structure.

This script helps migrate from the old flat structure to the new folder-based structure
and demonstrates how to organize characters with multiple reference images/videos.
"""

import sys
import os
from pathlib import Path
import shutil

def create_folder_structure():
    """Create example folder structure for multi-batch characters."""

    characters_path = Path("assets/characters")

    print("="*60)
    print("Multi-Batch Character Structure Setup")
    print("="*60)
    print()

    # Check if characters exist
    if not characters_path.exists():
        print(f"Creating {characters_path}...")
        characters_path.mkdir(parents=True, exist_ok=True)

    # Find existing character images
    existing_images = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        existing_images.extend(characters_path.glob(f"*{ext}"))

    if existing_images:
        print(f"Found {len(existing_images)} existing character images")
        print()
        print("Would you like to migrate them to folder structure? (y/n)")
        response = input("> ").strip().lower()

        if response == 'y':
            migrate_characters(existing_images, characters_path)
    else:
        print("No existing characters found.")
        print()
        create_example_structure(characters_path)

def migrate_characters(image_files, base_path):
    """Migrate flat character images to folder structure."""

    print()
    print("Migrating characters to folder structure...")
    print()

    for img_file in image_files:
        # Create folder for character
        char_name = img_file.stem
        char_folder = base_path / char_name

        if char_folder.exists():
            print(f"  Skipping {char_name} (folder already exists)")
            continue

        char_folder.mkdir(exist_ok=True)

        # Move image to folder
        new_path = char_folder / img_file.name
        shutil.move(str(img_file), str(new_path))

        print(f"  ✓ {char_name}/ (moved {img_file.name})")

    print()
    print("Migration complete!")
    print()
    print_usage_guide()

def create_example_structure(base_path):
    """Create example character folders."""

    print("Creating example structure...")
    print()

    # Create example folders
    examples = [
        "character1",
        "character2",
        "character3"
    ]

    for char_name in examples:
        char_folder = base_path / char_name
        char_folder.mkdir(exist_ok=True)

        # Create README in folder
        readme = char_folder / "README.txt"
        readme.write_text(
            f"Character: {char_name}\n\n"
            f"Add your reference materials here:\n"
            f"- Multiple images: reference1.png, reference2.jpg, etc.\n"
            f"- Multiple videos: video1.mp4, video2.mp4, etc.\n\n"
            f"Supported image formats: PNG, JPG, JPEG, BMP, WebP, TIFF\n"
            f"Supported video formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM\n\n"
            f"The system will:\n"
            f"- Extract frames from videos automatically\n"
            f"- Use all references to learn the character better\n"
            f"- Cache processed data for fast loading\n"
        )

        print(f"  ✓ Created {char_folder}/")

    print()
    print("Example structure created!")
    print()
    print_usage_guide()

def print_usage_guide():
    """Print usage instructions."""

    print("="*60)
    print("How to Use Multi-Batch Characters")
    print("="*60)
    print()
    print("1. Structure:")
    print("   assets/characters/")
    print("   ├── character1/")
    print("   │   ├── image1.png")
    print("   │   ├── image2.jpg")
    print("   │   ├── video1.mp4")
    print("   │   └── video2.mp4")
    print("   ├── character2/")
    print("   │   ├── reference.png")
    print("   │   └── expressions.mp4")
    print("   └── character3/")
    print("       └── main.png")
    print()
    print("2. Add your character files:")
    print("   - Put multiple reference images in each folder")
    print("   - Add videos with different expressions/angles")
    print("   - The system extracts frames automatically")
    print()
    print("3. Benefits:")
    print("   - Better character learning from multiple references")
    print("   - Videos provide diverse expressions/angles")
    print("   - Cached for fast loading")
    print("   - Automatic frame extraction and processing")
    print()
    print("4. Configuration (assets/config.yaml):")
    print("   character:")
    print("     enable_multi_batch: true")
    print("     enable_video_processing: true")
    print("     max_frames_per_video: 30")
    print("     video_sample_rate: 10")
    print()
    print("5. Run the application:")
    print("   python src/main.py --camera 1")
    print()
    print("="*60)
    print()

def check_current_structure():
    """Check and display current character structure."""

    characters_path = Path("assets/characters")

    if not characters_path.exists():
        print("No characters directory found.")
        return

    print("="*60)
    print("Current Character Structure")
    print("="*60)
    print()

    # Check for folders
    folders = [d for d in characters_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if folders:
        print(f"Found {len(folders)} character folders:")
        print()

        for folder in sorted(folders):
            images = []
            videos = []

            # Count images
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff']:
                images.extend(folder.glob(f"*{ext}"))
                images.extend(folder.glob(f"*{ext.upper()}"))

            # Count videos
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']:
                videos.extend(folder.glob(f"*{ext}"))
                videos.extend(folder.glob(f"*{ext.upper()}"))

            print(f"  {folder.name}/")
            print(f"    - {len(images)} image(s)")
            print(f"    - {len(videos)} video(s)")

            if images:
                for img in sorted(images)[:3]:  # Show first 3
                    print(f"      • {img.name}")
                if len(images) > 3:
                    print(f"      ... and {len(images)-3} more")

            if videos:
                for vid in sorted(videos)[:3]:  # Show first 3
                    print(f"      • {vid.name}")
                if len(videos) > 3:
                    print(f"      ... and {len(videos)-3} more")

            print()
    else:
        # Check for flat structure
        images = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            images.extend(characters_path.glob(f"*{ext}"))

        if images:
            print(f"Found {len(images)} images in flat structure (legacy mode)")
            print()
            print("Consider migrating to folder structure for multi-batch support.")
            print("Run this script to migrate.")
        else:
            print("No characters found.")

    print("="*60)
    print()

def main():
    """Main entry point."""

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'check':
            check_current_structure()
        elif command == 'setup':
            create_folder_structure()
        elif command == 'migrate':
            characters_path = Path("assets/characters")
            existing_images = []
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                existing_images.extend(characters_path.glob(f"*{ext}"))
            if existing_images:
                migrate_characters(existing_images, characters_path)
            else:
                print("No images to migrate.")
        elif command == 'help':
            print_usage_guide()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python setup_character_structure.py [check|setup|migrate|help]")
    else:
        # Interactive mode
        print("="*60)
        print("Multi-Batch Character Setup Tool")
        print("="*60)
        print()
        print("Options:")
        print("  1. Check current structure")
        print("  2. Setup new folder structure")
        print("  3. Migrate existing characters")
        print("  4. Show usage guide")
        print("  5. Exit")
        print()

        choice = input("Choose option (1-5): ").strip()

        if choice == '1':
            check_current_structure()
        elif choice == '2':
            create_folder_structure()
        elif choice == '3':
            characters_path = Path("assets/characters")
            existing_images = []
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                existing_images.extend(characters_path.glob(f"*{ext}"))
            if existing_images:
                migrate_characters(existing_images, characters_path)
            else:
                print("No images to migrate.")
        elif choice == '4':
            print_usage_guide()
        elif choice == '5':
            print("Goodbye!")
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()

