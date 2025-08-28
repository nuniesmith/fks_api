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
        # Test application import - try multiple possible locations
        app_found = False
        
        for module_name in ['fastapi_main', 'fks_api.fastapi_main', 'app', 'main']:
            try:
                module = __import__(module_name, fromlist=['app'] if '.' in module_name else [''])
                if hasattr(module, 'app'):
                    app = getattr(module, 'app')
                    print(f"✅ Application import successful (from {module_name})")
                    app_found = True
                    break
            except ImportError:
                continue
        
        if not app_found:
            raise ImportError("Could not find FastAPI app in any expected location")
            
    except ImportError as e:
        print(f"❌ Application import failed: {e}")
        return False
    
    try:
        # Test shared_python import (gracefully handle if not available)
        try:
            # Try importing something from shared_python
            import shared_python
            print("✅ Shared Python module accessible")
            # Try to import a specific function if it exists
            try:
                from shared_python import load_config
                print("✅ Shared Python load_config import successful")
            except (ImportError, AttributeError):
                print("⚠️  Shared Python load_config not available (this is okay)")
        except ImportError:
            print("⚠️  Shared Python module not available (using fallback - this is okay)")
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
