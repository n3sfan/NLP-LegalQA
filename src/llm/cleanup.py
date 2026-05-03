import os
from collections import defaultdict
from packaging import version
from packaging.utils import canonicalize_name

# Path to your folder
folder_path = "./offline_packages"

# Dictionary to store: { 'canonical_name': [ (version, full_filename), ... ] }
package_map = defaultdict(list)

# 1. Parse all files in the directory
for filename in os.listdir(folder_path):
    if not filename.endswith(".whl"):
        continue
    
    # Wheel naming convention: {distribution}-{version}-{build}-{python}-{abi}-{platform}.whl
    # We split by the first and second hyphens to get name and version
    parts = filename.split('-')
    if len(parts) < 2:
        continue
        
    raw_name = parts[0]
    ver_str = parts[1]
    
    # Canonicalize name (converts cuda_bindings and cuda-bindings to the same key)
    clean_name = canonicalize_name(raw_name)
    
    try:
        package_map[clean_name].append((version.parse(ver_str), filename))
    except Exception:
        # Skip files that don't follow standard versioning
        continue

# 2. Delete old versions
for name, file_list in package_map.items():
    if len(file_list) > 1:
        # Sort by version object (highest version last)
        file_list.sort(key=lambda x: x[0])
        
        # Keep the last one, delete all others
        newest_file = file_list[-1][1]
        for ver, old_filename in file_list[:-1]:
            print(f"Deleting older {name}: {old_filename} (Keeping {newest_file})")
            os.remove(os.path.join(folder_path, old_filename))

print("\nSuccess: Only the newest versions remain.")