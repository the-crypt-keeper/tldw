#!/usr/bin/env python3
"""Test runner with timeout for the ReDoS test."""

import sys
import signal
import subprocess
import time

def run_with_timeout(cmd, timeout_sec):
    """Run a command with timeout."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    try:
        stdout, _ = proc.communicate(timeout=timeout_sec)
        return proc.returncode, stdout
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        return -1, f"Test timed out after {timeout_sec} seconds\n{stdout}"

if __name__ == "__main__":
    # Run the fixed test with timeout
    cmd = [
        sys.executable, "-m", "pytest",
        "tldw_Server_API/tests/Chunking/test_security_fixed.py::TestReDoSProtection::test_complex_regex_timeout",
        "-xvs"
    ]
    
    print("Running test with 15-second timeout...")
    returncode, output = run_with_timeout(cmd, 15)
    
    print(output)
    
    if returncode == -1:
        print("\n❌ Test timed out - this suggests the ReDoS protection isn't working")
        sys.exit(1)
    elif returncode == 0:
        print("\n✅ Test passed!")
    else:
        print(f"\n❌ Test failed with return code {returncode}")
        sys.exit(returncode)