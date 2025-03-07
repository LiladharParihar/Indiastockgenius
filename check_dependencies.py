import sys
import pkg_resources
import subprocess
from typing import List, Tuple

REQUIRED_PACKAGES = {
    'tensorflow': '>=2.0.0',
    'numpy': '>=1.19.0',
    'pandas': '>=1.0.0',
    'yfinance': '>=0.2.0',
    'plotly': '>=5.0.0',
    'scikit-learn': '>=0.24.0',
    'PyQt5': '>=5.15.0',
    'PyQtWebEngine': '>=5.15.0',
    'pyttsx3': '>=2.90',
    'transformers': '>=4.0.0',
    'qdarkstyle': '>=3.0.0'
}

def check_package(package_name: str, required_version: str) -> Tuple[bool, str]:
    """Check if a package is installed and meets version requirements."""
    try:
        pkg_resources.require(f"{package_name}{required_version}")
        installed_version = pkg_resources.get_distribution(package_name).version
        return True, f"✓ {package_name} {installed_version}"
    except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound) as e:
        return False, f"✗ {str(e)}"

def install_package(package_name: str, version_spec: str) -> bool:
    """Install a package using pip."""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            f"{package_name}{version_spec}"
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependencies(auto_install: bool = False) -> bool:
    """
    Check all required dependencies and optionally install missing ones.
    
    Args:
        auto_install: If True, automatically install missing packages
    
    Returns:
        bool: True if all dependencies are satisfied, False otherwise
    """
    print("Checking dependencies...")
    print("-" * 50)
    
    all_satisfied = True
    missing_packages = []
    
    for package, version in REQUIRED_PACKAGES.items():
        satisfied, message = check_package(package, version)
        print(message)
        
        if not satisfied:
            all_satisfied = False
            missing_packages.append((package, version))
    
    if not all_satisfied and auto_install:
        print("\nInstalling missing packages...")
        for package, version in missing_packages:
            print(f"Installing {package}{version}...")
            if install_package(package, version):
                print(f"Successfully installed {package}")
            else:
                print(f"Failed to install {package}")
                return False
        
        # Recheck after installation
        print("\nRechecking dependencies...")
        all_satisfied = all(check_package(p, v)[0] for p, v in REQUIRED_PACKAGES.items())
    
    print("-" * 50)
    if all_satisfied:
        print("All dependencies are satisfied!")
    else:
        print("\nMissing dependencies. Please install them using:")
        for package, version in missing_packages:
            print(f"pip install {package}{version}")
    
    return all_satisfied

if __name__ == "__main__":
    # Check if --install flag is provided
    auto_install = "--install" in sys.argv
    if check_dependencies(auto_install):
        print("\nYou can now run the application using: python main.py")
    else:
        print("\nPlease install missing dependencies before running the application.")
        sys.exit(1) 