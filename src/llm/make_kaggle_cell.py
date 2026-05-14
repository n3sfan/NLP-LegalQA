import os

def generate():
    src_dir = 'src/llm'
    target_dir = 'llm'
    
    if not os.path.exists(src_dir):
        print(f"Error: Source directory {src_dir} not found.")
        return

    # Header for the notebook cell
    output = []
    output.append("import os")
    output.append(f"os.makedirs('{target_dir}', exist_ok=True)")

    # List and sort files for consistent output
    files = sorted([f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))])

    for filename in files:
        # Skip the generator itself if it's in the same folder
        if filename in ('generate_export.py', 'make_kaggle_cell.py'):
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

    # Print the final result to stdout
    print("\n".join(output))

if __name__ == "__main__":
    generate()
