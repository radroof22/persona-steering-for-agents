#!/usr/bin/env python3
"""
Check if the Sentosa environment is properly set up
"""

import sys
import subprocess

def check_conda_env():
    """Check if we're in the correct conda environment"""
    try:
        result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            current_env = None
            for line in lines:
                if line.strip().startswith('*'):
                    current_env = line.strip().split()[0]
                    break
            
            if current_env == 'sentosa':
                print("✅ Currently in 'sentosa' conda environment")
                return True
            else:
                print(f"❌ Not in 'sentosa' environment. Current: {current_env}")
                print("   Please activate the environment: 'mamba activate sentosa'")
                return False
    except Exception as e:
        print(f"❌ Could not check conda environment: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    dependencies = [
        ('pandas', 'pandas'),
        ('pyarrow', 'pyarrow'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('pytest', 'pytest'),
        ('requests', 'requests'),
        ('transformers', 'transformers'),
        ('torch', 'torch')
    ]
    
    missing = []
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {package_name} is available")
        except ImportError:
            print(f"❌ {package_name} is missing")
            missing.append(package_name)
    
    return missing

def main():
    """Main check function"""
    print("🔍 Checking Sentosa environment setup...")
    print("=" * 50)
    
    # Check conda environment
    env_ok = check_conda_env()
    
    print("\n📦 Checking dependencies...")
    missing_deps = check_dependencies()
    
    print("\n" + "=" * 50)
    
    if not env_ok:
        print("❌ Environment setup issues found:")
        print("   1. Activate the conda environment: 'mamba activate sentosa'")
        if missing_deps:
            print("   2. Install missing dependencies: 'mamba install " + " ".join(missing_deps) + "'")
        sys.exit(1)
    
    if missing_deps:
        print("❌ Missing dependencies:")
        print("   Install with: 'mamba install " + " ".join(missing_deps) + "'")
        sys.exit(1)
    
    print("✅ Environment is properly set up!")
    print("   You can now run: python main.py")

if __name__ == "__main__":
    main() 