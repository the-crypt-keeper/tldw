#!/usr/bin/env python3
"""
Fix for Gradio TypeError: argument of type 'bool' is not iterable
This error typically occurs due to version mismatches or config issues
"""

import subprocess
import sys

def fix_gradio_version():
    """Fix Gradio version issues"""
    print("Fixing Gradio version issues...")
    
    # First, uninstall existing gradio
    print("Uninstalling existing Gradio...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "gradio", "gradio-client"], check=False)
    
    # Install specific compatible versions
    print("Installing compatible Gradio version...")
    subprocess.run([sys.executable, "-m", "pip", "install", "gradio==4.44.1", "gradio-client==1.3.0"], check=True)
    
    print("Gradio version fix completed!")

def verify_fix():
    """Verify the fix worked"""
    try:
        import gradio as gr
        print(f"✅ Gradio version: {gr.__version__}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Gradio Error Fix Script")
    print("=" * 60)
    
    fix_gradio_version()
    
    if verify_fix():
        print("\n✅ Fix successful! Try running the app again.")
    else:
        print("\n❌ Fix failed. Please check the error messages above.")