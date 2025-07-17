#!/bin/bash

# Script to create conflict markers in patches/triton files,
# letting VSCode's built-in Git tools handle conflict resolution
# in your existing Git repository
# Usage: ./pull_upstream_triton.sh

# Initialize parameters and paths
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tilelink_root="$script_dir/.."
triton_path="$tilelink_root/3rdparty/triton"
conflict_list_file="$tilelink_root/conflict_list.txt"

# Function to generate conflict file directly in place
generate_conflict() {
    local patch_file="$1"
    local triton_file="$2"
    local temp_conflict_file=$(mktemp)
    
    # Generate conflict content to temp file
    python3 "$script_dir/diff_to_conflict.py" "$patch_file" "$triton_file" > "$temp_conflict_file"
    
    # Replace original patch file with conflict file
    cp -f "$temp_conflict_file" "$patch_file"
    rm -f "$temp_conflict_file"
    
    echo "Created conflict markers in: $patch_file"
}

# Check for git repository
if [ ! -d "$tilelink_root/.git" ]; then
    echo "Error: Not in a Git repository. This script needs to be run in a Git repository."
    exit 1
fi

# Check if working tree is clean
if ! git -C "$tilelink_root" diff --quiet; then
    echo "Warning: You have uncommitted changes in your working tree."
    echo "It's recommended to commit or stash your changes before running this script."
    echo "Do you want to continue anyway? [y/N]"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 0
    fi
fi

# Clear previous conflict list
> "$conflict_list_file"

# Detect differences and create conflict files
echo "Detecting and generating conflict files..."
conflict_count=0

for patch_file in $(find "$tilelink_root/patches/triton" -type f); do
    relative_path="${patch_file#$tilelink_root/patches/triton/}"
    triton_file="$triton_path/$relative_path"
    
    # Skip if triton file doesn't exist
    [ ! -f "$triton_file" ] && continue
    
    # Check for differences
    if ! diff -q "$patch_file" "$triton_file" > /dev/null; then
        echo "Detecting conflicts in: $relative_path"
        generate_conflict "$patch_file" "$triton_file"
        conflict_count=$((conflict_count + 1))
        echo "$patch_file" >> "$conflict_list_file"
    fi
done

# backend
for patch_file in $(find "$tilelink_root/backends" -type f); do
    relative_path="${patch_file#$tilelink_root/backends}"
    triton_file="$triton_path/third_party/$relative_path"
    
    # Skip if triton file doesn't exist
    [ ! -f "$triton_file" ] && continue
    
    echo "patch_file: $patch_file, triton_file: $triton_file"
    # Check for differences
    if ! diff -q "$patch_file" "$triton_file" > /dev/null; then
        echo "Detecting conflicts in: $relative_path"
        generate_conflict "$patch_file" "$triton_file"
        conflict_count=$((conflict_count + 1))
        echo "$patch_file" >> "$conflict_list_file"
    fi
done

# lib
dist_triton_ext_passes=("$tilelink_root/lib/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPU.cpp")
upstream_triton_passes=("$triton_path/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp")

for i in "${!dist_triton_ext_passes[@]}"; do

    patch_file=${dist_triton_ext_passes[$i]}
    triton_file=${upstream_triton_passes[$i]}
    # Skip if triton file doesn't exist
    [ ! -f "$triton_file" ] && continue

    echo "patch_file: $patch_file, triton_file: $triton_file"
    # Check for differences
    if ! diff -q "$patch_file" "$triton_file" > /dev/null; then
        echo "Detecting conflicts in: $relative_path"
        generate_conflict "$patch_file" "$triton_file"
        conflict_count=$((conflict_count + 1))
        echo "$patch_file" >> "$conflict_list_file"
    fi
done

# Exit if no conflicts found
if [ "$conflict_count" -eq 0 ]; then
    echo "No conflicts detected. Exiting."
    rm -f "$conflict_list_file"
    exit 0
fi

# Display instructions
echo -e "\n=== Conflict Resolution Instructions ==="
echo "Created $conflict_count files with conflict markers in patches/triton."
echo "List of conflict files:"
cat "$conflict_list_file"
echo ""
echo "To resolve conflicts:"
echo "1. Check VSCode's Source Control panel to see the modified files"
echo "2. Open each file and resolve conflicts using VSCode's conflict editor"
echo "3. Save the files when conflicts are resolved"
echo "4. After resolving all conflicts, you can stage and commit the changes"
echo ""
echo "To discard changes: Use Git commands like 'git restore' or VSCode's Source Control"
echo -e "\nHappy merging!"

# Function to check if all conflicts are resolved
check_conflicts_resolved() {
    if [ -f "$conflict_list_file" ]; then
        while read -r conflict_file; do
            if [ -f "$conflict_file" ] && (grep -q "<<<<<<<" "$conflict_file" || grep -q ">>>>>>>" "$conflict_file"); then
                return 1
            fi
        done < "$conflict_list_file"
    fi
    return 0
}

# Keep checking until all conflicts are resolved
while ! check_conflicts_resolved; do
    echo "There are still unresolved conflicts. Please resolve them before proceeding."
    read -p "Press enter to check again..."
done

echo "All conflicts are resolved. You can now stage and commit your changes."
rm -f "$conflict_list_file"

exit 0