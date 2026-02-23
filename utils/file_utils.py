"""
File I/O utilities with atomic writes and error handling.

Provides robust file operations to prevent data corruption
and handle common I/O errors gracefully.
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, Union
import logging

logger = logging.getLogger('helios.file_utils')


class FileWriteError(IOError):
    """Raised when file write operation fails."""
    pass


def atomic_write_json(data: Dict[str, Any], filepath: Union[str, Path],
                      indent: int = 2, **kwargs) -> None:
    """
    Write JSON file atomically to prevent corruption.

    Uses temporary file + atomic rename pattern to ensure
    the file is either fully written or not modified at all.

    Args:
        data: Data to write as JSON
        filepath: Target file path
        indent: JSON indentation (default: 2)
        **kwargs: Additional arguments passed to json.dump()

    Raises:
        FileWriteError: If write operation fails

    Example:
        >>> data = {'run_id': '20260102', 'results': [1, 2, 3]}
        >>> atomic_write_json(data, 'results.json')
    """
    filepath = Path(filepath)

    # Ensure directory exists
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise FileWriteError(f"Cannot create directory {filepath.parent}: {e}")

    # Write to temporary file in same directory
    # (Same directory ensures same filesystem for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=f'.{filepath.name}.',
        dir=filepath.parent
    )

    temp_path = Path(temp_path)

    try:
        # Write data to temp file
        with open(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, **kwargs)
            f.flush()
            import os
            os.fsync(f.fileno())  # Ensure written to disk

        # Atomic rename (POSIX-atomic on Unix, best-effort on Windows)
        temp_path.replace(filepath)

        logger.info(f"Successfully wrote {filepath}")

    except Exception as e:
        # Clean up temp file on error
        try:
            temp_path.unlink(missing_ok=True)
        except:
            pass

        raise FileWriteError(f"Failed to write {filepath}: {e}") from e


def atomic_write_text(text: str, filepath: Union[str, Path],
                      encoding: str = 'utf-8') -> None:
    """
    Write text file atomically.

    Args:
        text: Text content to write
        filepath: Target file path
        encoding: Text encoding (default: utf-8)

    Raises:
        FileWriteError: If write operation fails
    """
    filepath = Path(filepath)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise FileWriteError(f"Cannot create directory {filepath.parent}: {e}")

    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=f'.{filepath.name}.',
        dir=filepath.parent
    )

    temp_path = Path(temp_path)

    try:
        with open(temp_fd, 'w', encoding=encoding) as f:
            f.write(text)
            f.flush()
            import os
            os.fsync(f.fileno())

        temp_path.replace(filepath)
        logger.info(f"Successfully wrote {filepath}")

    except Exception as e:
        try:
            temp_path.unlink(missing_ok=True)
        except:
            pass

        raise FileWriteError(f"Failed to write {filepath}: {e}") from e


def safe_read_json(filepath: Union[str, Path], default: Any = None) -> Union[Dict, Any]:
    """
    Read JSON file with error handling.

    Args:
        filepath: Path to JSON file
        default: Default value if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or default value

    Example:
        >>> data = safe_read_json('config.json', default={})
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning(f"File not found: {filepath}, returning default")
        return default

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}, returning default")
        return default
    except IOError as e:
        logger.error(f"Cannot read {filepath}: {e}, returning default")
        return default


def ensure_directory(dirpath: Union[str, Path], mode: int = 0o755) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        dirpath: Directory path
        mode: Permission mode for new directories

    Returns:
        Path object for the directory

    Raises:
        FileWriteError: If directory cannot be created
    """
    dirpath = Path(dirpath)

    try:
        dirpath.mkdir(parents=True, exist_ok=True, mode=mode)
        return dirpath
    except OSError as e:
        raise FileWriteError(f"Cannot create directory {dirpath}: {e}") from e


def check_disk_space(filepath: Union[str, Path], required_mb: float = 10.0) -> bool:
    """
    Check if sufficient disk space is available.

    Args:
        filepath: Path to check (uses parent directory)
        required_mb: Required space in megabytes

    Returns:
        True if sufficient space available

    Example:
        >>> if not check_disk_space('output.json', required_mb=100):
        ...     logger.warning("Low disk space!")
    """
    filepath = Path(filepath)

    try:
        import shutil
        stat = shutil.disk_usage(filepath.parent if filepath.parent.exists() else '.')
        available_mb = stat.free / (1024 * 1024)

        if available_mb < required_mb:
            logger.warning(f"Low disk space: {available_mb:.1f}MB available, "
                          f"{required_mb:.1f}MB required")
            return False

        return True

    except Exception as e:
        logger.error(f"Cannot check disk space: {e}")
        return True  # Assume sufficient if we can't check


def backup_file(filepath: Union[str, Path], backup_suffix: str = '.bak') -> Optional[Path]:
    """
    Create backup of existing file.

    Args:
        filepath: File to backup
        backup_suffix: Suffix for backup file

    Returns:
        Path to backup file or None if original doesn't exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return None

    backup_path = filepath.with_suffix(filepath.suffix + backup_suffix)

    try:
        shutil.copy2(filepath, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None


if __name__ == "__main__":
    # Demo/test
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing atomic file writes...")

    # Test atomic JSON write
    test_data = {
        'timestamp': '2026-01-02',
        'values': [1, 2, 3, 4, 5],
        'metadata': {'version': '1.0'}
    }

    test_dir = Path('test_output')
    test_file = test_dir / 'test_atomic.json'

    try:
        atomic_write_json(test_data, test_file)
        print(f"✓ Wrote {test_file}")

        # Test read back
        loaded = safe_read_json(test_file)
        assert loaded == test_data, "Data mismatch!"
        print(f"✓ Read back matches")

        # Test backup
        backup = backup_file(test_file)
        print(f"✓ Created backup: {backup}")

        # Cleanup
        test_file.unlink()
        if backup:
            backup.unlink()
        test_dir.rmdir()

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
