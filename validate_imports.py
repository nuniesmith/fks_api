#!/usr/bin/env python3
"""
Validation script to check that all imports work correctly.
This can be run in CI to catch import issues early.
"""

import sys
import os

# Add the same paths as the application
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared_python'))

def validate_imports():
    """Validate that critical imports work."""
    print("🔍 Validating imports...")
    
    try:
        # Test basic imports
        import fastapi
        print("✅ FastAPI import successful")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
        
    try:
        # Test application import
        from fks_api.fastapi_main import app
        print("✅ Application import successful")
    except ImportError as e:
        print(f"❌ Application import failed: {e}")
        return False
    
    try:
        # Test shared_python import (should not fail even if not available)
        try:
            from shared_python import load_config
            print("✅ Shared Python import successful")
        except ImportError:
            print("⚠️  Shared Python import not available (using fallback)")
    except Exception as e:
        print(f"❌ Unexpected error with shared_python: {e}")
        return False
    
    print("🎉 All imports validated successfully!")
    return True

if __name__ == "__main__":
    if validate_imports():
        sys.exit(0)
    else:
        sys.exit(1)
