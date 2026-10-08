#!/usr/bin/env python3
"""Fix URDF orientations by removing rpy rotations from origin tags."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def fix_urdf(urdf_path: Path, backup: bool = True) -> bool:
    """Remove rpy rotations from URDF origin tags."""
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        modified = False
        for origin in root.findall(".//origin"):
            rpy = origin.get("rpy")
            if rpy and rpy != "0.0 0.0 0.0" and rpy != "0 0 0":
                print(f"  Found rotation: rpy='{rpy}' -> setting to '0.0 0.0 0.0'")
                origin.set("rpy", "0.0 0.0 0.0")
                modified = True
        
        if modified:
            if backup:
                backup_path = urdf_path.with_suffix(".urdf.bak")
                urdf_path.rename(backup_path)
                print(f"  Backed up to: {backup_path}")
            
            # Write with proper formatting
            ET.indent(tree, space="   ")
            tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
            print(f"  ✓ Fixed: {urdf_path}")
            return True
        else:
            print(f"  No changes needed: {urdf_path}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error processing {urdf_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fix URDF orientations in custom assets")
    parser.add_argument(
        "--assets-root",
        type=str,
        default="our_assets/actor",
        help="Root directory containing custom assets"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files"
    )
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parents[1]
    assets_root = (repo_root / args.assets_root).resolve()
    
    if not assets_root.exists():
        print(f"Error: Assets root not found: {assets_root}")
        return 1
    
    print(f"Scanning: {assets_root}")
    print()
    
    urdf_files = list(assets_root.rglob("*.urdf"))
    if not urdf_files:
        print("No URDF files found.")
        return 0
    
    print(f"Found {len(urdf_files)} URDF file(s)\n")
    
    fixed_count = 0
    for urdf_path in sorted(urdf_files):
        print(f"Processing: {urdf_path.relative_to(assets_root)}")
        if fix_urdf(urdf_path, backup=not args.no_backup):
            fixed_count += 1
        print()
    
    print(f"Summary: Fixed {fixed_count} of {len(urdf_files)} URDFs")
    return 0


if __name__ == "__main__":
    exit(main())
