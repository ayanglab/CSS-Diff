import os
import re

def rename_images(root_dir, apply=False):
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(root_dir)
    operations = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip if the directory doesn't end with 'A' or 'B'
        if not (dirpath.endswith("A") or dirpath.endswith("B")):
            continue

        # Decide the prefix
        if dirpath.endswith("A"):
            prefix = "low_field"
        elif dirpath.endswith("B"):
            prefix = "high_field"
        else:
            continue

        print(f"Renaming files in: {dirpath} -> prefix: {prefix}")

        for fname in sorted(filenames):
            if not fname.endswith(".png"):
                continue

            # Extract numerical part from filename (assumes filename is like 0000.png)
            base, ext = os.path.splitext(fname)
            try:
                number = int(base)
            except ValueError:
                print(f"Skipping {fname}, not a valid integer name.")
                continue

            new_name = f"{prefix}_{number}{ext}"
            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(dirpath, new_name)

            print(f"Renaming: {fname} -> {new_name}")
            if os.path.exists(dst_path):
                raise FileExistsError(dst_path)
            operations.append((src_path, dst_path))
    if apply:
        for src, dst in operations:
            os.rename(src, dst)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preview numeric MRI slice renaming; use --apply to rename.')
    parser.add_argument('root_dir')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    rename_images(args.root_dir, apply=args.apply)
