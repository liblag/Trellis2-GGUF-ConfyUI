import os
import sys
import subprocess
import platform
import argparse
import shutil
import site

def run_command(command):
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    return True

def is_uv_available():
    try:
        subprocess.check_call([sys.executable, "-m", "uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def get_env_info():
    info = {}
    
    # OS
    if platform.system() == "Windows":
        info['platform'] = "win_amd64"
    else:
        info['platform'] = "manylinux_2_35_x86_64" # Default specialized tag
        
    # Python version
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    info['python'] = py_ver
    
    # Torch and CUDA
    try:
        import torch
        torch_raw = torch.__version__.split('+')[0]
        torch_ver = torch_raw.split('.')
        info['torch'] = f"torch{torch_ver[0]}{torch_ver[1]}"
        info['torch_raw'] = torch_raw
        
        if torch.version.cuda:
            cuda_ver = torch.version.cuda.split('.')
            info['cuda'] = f"cu{cuda_ver[0]}{cuda_ver[1]}"
        else:
            info['cuda'] = None
    except ImportError:
        print("PyTorch not found. Please install PyTorch with CUDA support first.")
        sys.exit(1)
        
    return info

def apply_patches(dry_run=False):
    print("\n--- Applying Patches ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    patch_dir = os.path.join(current_dir, "patch")
    
    if not os.path.exists(patch_dir):
        print(f"Patch folder not found at {patch_dir}. Skipping patching.")
        return

    # Mapping: filename in patch/ -> target library name
    patch_mapping = {
        "remeshing.py": "cumesh",
        "flexible_dual_grid.py": "o_voxel"
    }
    
    import importlib.util

    for patch_file, lib_name in patch_mapping.items():
        src = os.path.join(patch_dir, patch_file)
        if not os.path.exists(src):
            continue

        print(f"Searching for target of {patch_file} in {lib_name}...")
        
        lib_path = None
        try:
            spec = importlib.util.find_spec(lib_name)
            if spec and spec.origin:
                lib_path = os.path.dirname(spec.origin)
        except Exception as e:
            print(f"  Error searching for {lib_name}: {e}")

        if not lib_path:
            print(f"  Warning: Library '{lib_name}' not found. Skipping patch {patch_file}.")
            continue

        # Environment-independent search within the library directory
        target_path = None
        for root, dirs, files in os.walk(lib_path):
            if patch_file in files:
                target_path = os.path.join(root, patch_file)
                break
        
        if target_path:
            if dry_run:
                print(f"  [Dry Run] Would patch: {src} -> {target_path}")
            else:
                try:
                    shutil.copy(src, target_path)
                    print(f"  Successfully patched: {target_path}")
                except Exception as e:
                    print(f"  Error patching {target_path}: {e}")
        else:
            print(f"  Warning: Could not find '{patch_file}' within {lib_path}. Library might be a different version.")

def show_recommendations():
    print("\n" + "="*50)
    print("RECOMMENDED ENVIRONMENTS")
    print("="*50)
    print("If you encounter 404 errors, it is likely because your environment")
    print("combination is not yet supported on the wheel repository.")
    print("\nPreferred configurations:")
    print("1. CUDA 12.4 + PyTorch 2.5 (Most compatible)")
    print("2. CUDA 12.6 + PyTorch 2.6 (Latest compatible)")
    print("3. CUDA 12.1 + PyTorch 2.4 (Legacy compatible)")
    print("\nYou can install the preferred Torch version using:")
    print("pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124")
    print("="*50 + "\n")

def find_wheel_url(lib_name, env):
    import urllib.request
    import re
    
    index_url = f"https://pozzettiandrea.github.io/cuda-wheels/{lib_name}/"
    
    try:
        req = urllib.request.Request(index_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
    except Exception as e:
        print(f"  Warning: Could not access {lib_name} wheel index: {e}")
        return None

    cuda = env['cuda']
    torch_tag = env['torch']
    python_tag = env['python']
    
    # torch_tag is like 'torch210'
    torch_base = torch_tag[:5] # 'torch'
    torch_ver = torch_tag[5:] # '210'
    
    v_major = torch_ver[0]
    v_minor = torch_ver[1:]
    
    # 1. Handle encoding: '+' in filenames is often '%2B' in HTML URLs
    # 2. Handle versioning: Filenames can have 'torch2.10' or 'torch210'
    # 3. Handle platform: Use a wildcard for the platform part to be robust
    
    # We escape the dots and plus for regex, but also allow the encoded version
    # The filename part usually looks like: {lib_name}-{version}+{cuda}torch{torch_ver}-{python}-{python}-{platform}.whl
    
    # Regex explanation:
    # - Matches the package name (with underscores instead of hyphens)
    # - Matches the version block and a '+' or '%2B'
    # - Matches the cuda string (e.g. cu128)
    # - Matches the torch string (e.g. torch210 or torch2.10)
    # - Matches the python tags (e.g. -cp311-cp311-)
    # - Wildcard for the platform tag and extension
    
    lib_name_fixed = lib_name.replace("-", "_")
    pattern = rf'https://github\.com/[^"\s]+{lib_name_fixed}-[^"\s]+(?:\+|%2B){cuda}{torch_base}{v_major}\.?{v_minor}-{python_tag}-{python_tag}-[^"\s]+\.whl'
    
    matches = re.findall(pattern, html)
    if matches:
        return matches[0]

    return None

def install_flash_attn(pip_base, env, dry_run=False):
    print("\n--- Installing flash-attn ---")
    wheel_url = find_wheel_url("flash-attn", env)
    
    if wheel_url:
        print(f"  Found matching wheel: {wheel_url}")
        if dry_run:
            print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', wheel_url])}")
            return True
        else:
            return run_command(pip_base + ["install", wheel_url])
    else:
        print(f"  Warning: No matching flash-attn wheel found for {env['cuda']}, {env['torch']}, {env['python']} on {env['platform']}.")
        print("  Please consider switching to a recommended environment for flash-attn support.")
        return False

def install_triton_windows(pip_base, env, dry_run=False):
    """
    Installs the correct version of triton-windows based on the PyTorch version 
    to prevent 'cluster_dims' or other metadata crashes.
    """
    if platform.system() != "Windows":
        return True

    print("\n--- Installing triton-windows ---")
    
    torch_raw = env.get('torch_raw', '2.0.0')
    try:
        from packaging import version
    except ImportError:
        # Fallback if packaging isn't installed for some reason
        class VersionFallback:
            def __init__(self, v): self.v = [int(x) for x in v.split('.')[:2]]
            def __ge__(self, other): return self.v >= [int(x) for x in other.split('.')[:2]]
        version = type('obj', (object,), {'parse': lambda v: VersionFallback(v)})

    v_torch = version.parse(torch_raw)
    triton_spec = "triton-windows"
    
    if v_torch >= version.parse("2.10.0"):
        triton_spec = "triton-windows<3.7"
        print("  Detected Torch >= 2.10. Using triton-windows < 3.7 (v3.6.0)")
    elif v_torch >= version.parse("2.9.0"):
        triton_spec = "triton-windows<3.6"
        print("  Detected Torch >= 2.9. Using triton-windows < 3.6 (v3.5.1)")
    elif v_torch >= version.parse("2.8.0"):
        triton_spec = "triton-windows<3.5"
        print("  Detected Torch >= 2.8. Using triton-windows < 3.5 (v3.4.0)")
    elif v_torch >= version.parse("2.7.0"):
        triton_spec = "triton-windows<3.4"
        print("  Detected Torch >= 2.7. Using triton-windows < 3.4 (v3.3.1)")
    elif v_torch >= version.parse("2.6.0"):
        triton_spec = "triton-windows<3.3"
        print("  Detected Torch >= 2.6. Using triton-windows < 3.3 (v3.2.0)")
    else:
        print("  Torch < 2.6 detected. Using default triton-windows.")

    if dry_run:
        print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', '-U', triton_spec])}")
        return True
    else:
        return run_command(pip_base + ["install", "-U", triton_spec])

def install():
    parser = argparse.ArgumentParser(description="ComfyUI-Trellis2 Installation")
    parser.add_argument("--dry-run", action="store_true", help="Print the steps without installing")
    args = parser.parse_args()

    print("--- ComfyUI-Trellis2 Installation ---")
    if args.dry_run:
        print("(DRY RUN MODE)")
    
    env = get_env_info()
    print(f"Detected Environment:")
    print(f"  OS: {platform.system()}")
    print(f"  Python: {env['python']}")
    print(f"  PyTorch: {env['torch']}")
    print(f"  CUDA: {env['cuda']}")
    
    if not env['cuda']:
        print("Error: CUDA not detected in PyTorch. These wheels require CUDA.")
        sys.exit(1)
    
    # 1. Install standard requirements
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, "requirements.txt")
    
    use_uv = is_uv_available()
    pip_base = [sys.executable, "-m", "uv", "pip"] if use_uv else [sys.executable, "-m", "pip"]
    
    if os.path.exists(requirements_path):
        print(f"\nInstalling requirements from {requirements_path}...")
        if args.dry_run:
            print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', '-r', requirements_path])}")
        else:
            run_command(pip_base + ["install", "-r", requirements_path])

    # 2. Install flash-attn from Pozzetti's wheels
    install_flash_attn(pip_base, env, dry_run=args.dry_run)
    
    # 3. Install correct triton-windows version
    install_triton_windows(pip_base, env, dry_run=args.dry_run)
    
    # 3. Define other CUDA wheels
    packages = [
        {"name": "cumesh", "tag": "cumesh-latest", "file": "cumesh", "version": "0.0.1", "linux_tag": "manylinux_2_35_x86_64"},
        {"name": "flex-gemm", "tag": "flex_gemm-latest", "file": "flex_gemm", "version": "1.0.0", "linux_tag": "manylinux_2_34_x86_64.manylinux_2_35_x86_64"},
        {"name": "nvdiffrast", "tag": "nvdiffrast-latest", "file": "nvdiffrast", "version": "0.4.0", "linux_tag": "manylinux_2_34_x86_64.manylinux_2_35_x86_64"},
        {"name": "nvdiffrec-render", "tag": "nvdiffrec_render-latest", "file": "nvdiffrec_render", "version": "0.0.1", "linux_tag": "manylinux_2_34_x86_64.manylinux_2_35_x86_64"},
        {"name": "o-voxel", "tag": "o_voxel-latest", "file": "o_voxel", "version": "0.0.1", "linux_tag": "manylinux_2_34_x86_64.manylinux_2_35_x86_64"},
    ]
    
    print("\nInstalling CUDA wheels...")
    
    errors_occurred = False
    for pkg in packages:
        print(f"Installing {pkg['name']}...")
        
        wheel_url = find_wheel_url(pkg['name'], env)
        
        if wheel_url:
            print(f"  Found matching wheel: {wheel_url}")
            if args.dry_run:
                print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', wheel_url])}")
            else:
                if not run_command(pip_base + ["install", wheel_url]):
                    print(f"Warning: Failed to install {pkg['name']}.")
                    errors_occurred = True
        else:
            print(f"  Warning: No matching {pkg['name']} wheel found in the index.")
            errors_occurred = True

    if errors_occurred:
        show_recommendations()

    # Apply Patches
    apply_patches(dry_run=args.dry_run)

    # 4. Install Smart-UV-Projection (pure Python, universal wheel)
    smart_uv_url = "https://github.com/Aero-Ex/Smart-UV-Projection/releases/download/v0.1.0/smart_uv_projection-0.1.0-py3-none-any.whl"
    print(f"\nInstalling Smart-UV-Projection...")
    if args.dry_run:
        print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', smart_uv_url])}")
    else:
        if not run_command(pip_base + ["install", smart_uv_url]):
            print("Warning: Failed to install Smart-UV-Projection. The 'Smart' UV unwrap method will not be available.")

    print("\nInstallation complete!")

if __name__ == "__main__":
    install()
