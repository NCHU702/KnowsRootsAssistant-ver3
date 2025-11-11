"""
PDF Storage Module

Handles PDF file storage with filename sanitization and duplicate handling.
"""

import os
import shutil
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default storage directory
DEFAULT_STORAGE_DIR = "./data"

# Maximum filename length (excluding extension)
MAX_FILENAME_LENGTH = 200


class StorageError(Exception):
    """Custom exception for storage errors"""
    pass


def sanitize_filename(title: str) -> str:
    """
    Sanitize paper title for use as filename
    
    Args:
        title: Paper title
        
    Returns:
        Sanitized filename (without extension)
    """
    # Remove or replace invalid characters
    # Windows: < > : " / \ | ? *
    # Also remove other problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', title)
    
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length
    if len(sanitized) > MAX_FILENAME_LENGTH:
        sanitized = sanitized[:MAX_FILENAME_LENGTH].strip()
    
    # If empty after sanitization, use default name
    if not sanitized:
        sanitized = "untitled_paper"
    
    return sanitized


def get_unique_filename(directory: str, base_filename: str, extension: str = ".pdf") -> str:
    """
    Get unique filename by appending counter if needed
    
    Args:
        directory: Target directory
        base_filename: Base filename (without extension)
        extension: File extension (default: .pdf)
        
    Returns:
        Unique filename (without directory path)
    """
    filename = base_filename + extension
    filepath = os.path.join(directory, filename)
    
    # If file doesn't exist, use it
    if not os.path.exists(filepath):
        return filename
    
    # Otherwise, try with counter suffix
    counter = 2
    while counter < 1000:  # Safety limit
        filename = f"{base_filename}_{counter}{extension}"
        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            return filename
        
        counter += 1
    
    # If we reach here, something is wrong
    raise StorageError(f"Cannot create unique filename for: {base_filename}")


def store_pdf(source_path: str, paper_title: str, storage_dir: str = DEFAULT_STORAGE_DIR) -> str:
    """
    Store PDF file in data directory with title-based naming
    
    Args:
        source_path: Path to source PDF file
        paper_title: Title of the paper
        storage_dir: Target storage directory
        
    Returns:
        Final filename (without directory path)
        
    Raises:
        StorageError: If storage fails
    """
    try:
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Sanitize title for filename
        base_filename = sanitize_filename(paper_title)
        
        # Get unique filename
        filename = get_unique_filename(storage_dir, base_filename, ".pdf")
        
        # Full target path
        target_path = os.path.join(storage_dir, filename)
        
        # Copy file
        shutil.copy2(source_path, target_path)
        
        logger.info(f"PDF stored successfully: {filename}")
        return filename
    
    except OSError as e:
        logger.error(f"Storage failed: {e}")
        if "space" in str(e).lower():
            raise StorageError("Insufficient disk space")
        elif "permission" in str(e).lower():
            raise StorageError("Permission denied to write to storage directory")
        else:
            raise StorageError(f"Failed to store PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected storage error: {e}")
        raise StorageError(f"Failed to store PDF: {str(e)}")


def delete_temp_file(file_path: str):
    """
    Delete temporary file
    
    Args:
        file_path: Path to temporary file
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Deleted temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to delete temporary file {file_path}: {e}")


class PDFStorage:
    """Manages PDF file storage"""
    
    def __init__(self, storage_dir: str = DEFAULT_STORAGE_DIR):
        """
        Initialize PDF storage manager
        
        Args:
            storage_dir: Directory for storing PDFs
        """
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def store(self, source_path: str, paper_title: str) -> str:
        """
        Store PDF file
        
        Args:
            source_path: Path to source PDF
            paper_title: Paper title for filename
            
        Returns:
            Final filename
        """
        return store_pdf(source_path, paper_title, self.storage_dir)
    
    def cleanup_temp(self, temp_path: str):
        """
        Clean up temporary file
        
        Args:
            temp_path: Path to temporary file
        """
        delete_temp_file(temp_path)
    
    def get_storage_path(self, filename: str) -> str:
        """
        Get full path for stored file
        
        Args:
            filename: Filename
            
        Returns:
            Full path
        """
        return os.path.join(self.storage_dir, filename)
    
    def file_exists(self, filename: str) -> bool:
        """
        Check if file exists in storage
        
        Args:
            filename: Filename to check
            
        Returns:
            True if exists
        """
        return os.path.exists(self.get_storage_path(filename))
