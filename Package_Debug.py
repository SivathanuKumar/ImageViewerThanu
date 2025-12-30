import sys
import platform
import os
import importlib.metadata


def check_package_version(package_name):
    try:
        version = importlib.metadata.version(package_name)
        print(f"✅ {package_name}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"❌ {package_name}: Not Installed")


def check_system_info():
    print("=" * 40)
    print("SYSTEM INFORMATION")
    print("=" * 40)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"OS Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor Info: {platform.processor()}")

    # Check for specific DLL environment variables
    print("\nCUDA/PATH Check:")
    path_dirs = os.environ.get('PATH', '').split(';')
    cuda_paths = [p for p in path_dirs if 'NVIDIA' in p or 'CUDA' in p]
    if cuda_paths:
        for p in cuda_paths:
            print(f"  Found CUDA Path: {p}")
    else:
        print("  No obvious CUDA/NVIDIA paths found in PATH variable.")


def check_critical_libs():
    print("\n" + "=" * 40)
    print("CRITICAL LIBRARIES VERSIONS")
    print("=" * 40)
    # These are the most common culprits for DLL failures
    packages = [
        "tensorflow",
        "tensorflow-intel",
        "tensorflow-gpu",
        "numpy",
        "protobuf",
        "h5py",
        "keras"
    ]

    for pkg in packages:
        check_package_version(pkg)


if __name__ == "__main__":
    try:
        check_system_info()
        check_critical_libs()

        print("\n" + "=" * 40)
        print("ATTEMPTING TENSORFLOW IMPORT")
        print("=" * 40)
        import tensorflow as tf

        print(f"SUCCESS! TensorFlow {tf.__version__} imported correctly.")
        print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

    except Exception as e:
        print("\n💥 IMPORT FAILED AS EXPECTED")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")