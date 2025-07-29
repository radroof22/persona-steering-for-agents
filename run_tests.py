#!/usr/bin/env python3
"""
Test runner script for Sentosa API
"""

import subprocess
import sys
import argparse

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Sentosa API tests")
    parser.add_argument("--test-type", choices=["all", "quick", "query", "comprehensive"], 
                       default="quick", help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-s", action="store_true", help="Show output during tests")
    
    args = parser.parse_args()
    
    # Build pytest command
    pytest_args = ["pytest"]
    
    if args.verbose:
        pytest_args.append("-v")
    
    if args.output:
        pytest_args.append("-s")
    
    # Select test file based on type
    if args.test_type == "quick":
        pytest_args.append("test_query_api.py")
        description = "Running Quick Tests (Query API only)"
    elif args.test_type == "query":
        pytest_args.append("test_query_api.py")
        description = "Running Query API Tests"
    elif args.test_type == "comprehensive":
        pytest_args.append("test_api.py")
        description = "Running Comprehensive Test Suite"
    else:  # all
        pytest_args.extend(["test_query_api.py", "test_api.py"])
        description = "Running All Tests"
    
    cmd = " ".join(pytest_args)
    
    print("🚀 Sentosa API Test Runner")
    print(f"Test type: {args.test_type}")
    print(f"Command: {cmd}")
    
    success = run_command(cmd, description)
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 