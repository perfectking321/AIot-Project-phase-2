#!/usr/bin/env python3
"""
Quick Chrome Remote Debugging Check
"""

import requests
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

def check_chrome_debugging():
    print("Checking for Chrome with remote debugging...")

    found = False
    for port in [9222, 9223, 9224, 9225]:
        try:
            print(f"  Checking port {port}...")
            response = requests.get(f"http://localhost:{port}/json/version", timeout=3)
            if response.status_code == 200:
                data = response.json()
                print(f"  [OK] Found Chrome on port {port}")
                print(f"     Browser: {data.get('Browser', 'Unknown')}")
                print(f"     User-Agent: {data.get('User-Agent', 'Unknown')}")
                found = True
                break
        except Exception as e:
            print(f"  [X] Port {port}: Connection refused")

    if not found:
        print("\n[X] Chrome not found with remote debugging!")
        print("\nTo fix this:")
        print("1. Close all Chrome windows")
        print("2. Run this command:")
        print('   "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222')
        print("3. Or use the batch file: start_chrome_for_voxcode.bat")
        return False
    else:
        print("\n[OK] Chrome is ready for VOXCODE!")
        return True

if __name__ == "__main__":
    check_chrome_debugging()