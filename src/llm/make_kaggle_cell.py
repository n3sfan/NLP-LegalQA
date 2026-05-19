import os
import sys

def generate():
    src_dir = 'src/llm'
    target_dir = 'llm'
    include_dirs = ['prompts']
    
    if not os.path.exists(src_dir):
        print(f"Error: Source directory {src_dir} not found.")
        return

    # Header for the notebook cell
    output = []
    output.append("import os")
    output.append(f"os.makedirs('{target_dir}', exist_ok=True)")
    for dirname in include_dirs:
        output.append(f"os.makedirs('{target_dir}/{dirname}', exist_ok=True)")

    # List and sort files for consistent output
    files = sorted([f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))])

    for filename in files:
        # Skip the generator itself if it's in the same folder
        if filename in ('generate_export.py', 'make_kaggle_cell.py', 'cellcode.py'):
            continue
            
        path = os.path.join(src_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            output.append(f"\nprint('Writing {target_dir}/{filename}...')")
            output.append(f"with open('{target_dir}/{filename}', 'w', encoding='utf-8') as f:")
            output.append(f"    f.write({repr(content)})")
        except Exception as e:
            output.append(f"\n# Error reading {filename}: {e}")

    for dirname in include_dirs:
        dir_path = os.path.join(src_dir, dirname)
        if not os.path.isdir(dir_path):
            output.append(f"\n# Warning: Directory {dir_path} not found.")
            continue

        nested_files = sorted(
            f for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
        )

        for filename in nested_files:
            path = os.path.join(dir_path, filename)
            target_path = f"{target_dir}/{dirname}/{filename}"
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                output.append(f"\nprint('Writing {target_path}...')")
                output.append(f"with open('{target_path}', 'w', encoding='utf-8') as f:")
                output.append(f"    f.write({repr(content)})")
            except Exception as e:
                output.append(f"\n# Error reading {dirname}/{filename}: {e}")

    # Print the final result to stdout
    with open('src/llm/cellcode.py', 'w', encoding='utf-8') as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
    generate()
