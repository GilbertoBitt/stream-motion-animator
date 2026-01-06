"""
Tool to preprocess all character images for optimized inference.

Pre-computes and caches image features to speed up runtime performance.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from character_manager import CharacterManager
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_characters(config_path: str = None):
    """
    Preprocess all character images.

    Args:
        config_path: Path to config file
    """
    logger.info("=" * 60)
    logger.info("Character Image Preprocessor")
    logger.info("=" * 60)

    # Load config
    config = load_config(config_path)

    # Initialize character manager with preprocessing enabled
    logger.info(f"Loading characters from: {config.characters_path}")

    target_size_list = config.get('character.target_size', [512, 512])
    character_manager = CharacterManager(
        characters_path=config.characters_path,
        target_size=(int(target_size_list[0]), int(target_size_list[1])),
        auto_crop=config.get('character.auto_crop', True),
        preload_all=True,
        use_preprocessing_cache=True
    )

    # Get info
    info = character_manager.get_info()

    logger.info("")
    logger.info("Preprocessing Complete!")
    logger.info("-" * 60)
    logger.info(f"Characters processed: {info['character_count']}")
    logger.info(f"Target size: {info['target_size']}")

    if 'cache_stats' in info:
        cache_stats = info['cache_stats']
        logger.info(f"Cache directory: {cache_stats['cache_dir']}")
        logger.info(f"Disk cache entries: {cache_stats['disk_cache_count']}")
        logger.info(f"Memory cache entries: {cache_stats['memory_cache_count']}")
        logger.info(f"Memory usage: {cache_stats['memory_usage_mb']:.2f} MB")

    logger.info("-" * 60)
    logger.info("Preprocessed data will be used for faster inference!")
    logger.info("Run the application normally - it will use the cached data.")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess character images")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )

    args = parser.parse_args()

    try:
        preprocess_characters(args.config)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

