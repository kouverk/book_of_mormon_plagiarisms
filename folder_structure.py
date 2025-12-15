import os

def generate_tree(folder_path, prefix=""):
    lines = []

    base = os.path.basename(os.path.abspath(folder_path))
    lines.append(f"{prefix}/{base}")

    def walk(path, level):
        entries = sorted(os.listdir(path))
        for entry in entries:
            full_path = os.path.join(path, entry)

            # Skip any folder named 'bin' or 'venv'
            if os.path.isdir(full_path) and os.path.basename(full_path) in ("bin", "venv"):
                continue

            indent = "--" * (level + 1)
            if os.path.isfile(full_path):
                lines.append(f"{indent}{entry}")
            elif os.path.isdir(full_path):
                lines.append(f"{indent}/{entry}")
                walk(full_path, level + 1)

    walk(folder_path, 0)
    return "\n".join(lines)

if __name__ == "__main__":
    folder = "."  # current directory
    tree_output = generate_tree(folder)

    with open("folder_structure.txt", "w") as f:
        f.write(tree_output)

    print("✅ Folder structure written to folder_structure.txt (excluding 'bin' and 'venv')")
