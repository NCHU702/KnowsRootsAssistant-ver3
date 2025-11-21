#!/usr/bin/env python3
"""Fix Layer1 indentation issues in hierarchical_rag_system.py"""

file_path = "/Users/charles88/Desktop/KnowsRootsAssistant-ver3/system_api/hierarchical_rag_system.py"

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the problematic section
# Lines 909-1054 need to be indented by 4 spaces (one level)
start_line = 909 - 1  # 0-indexed
end_line = 1054 - 1   # Up to the else clause

# Add indentation to these lines (except if they're already comments or empty)
for i in range(start_line, end_line):
    if i < len(lines):
        line = lines[i]
        # Add 4 spaces indentation if not empty
        if line.strip():
            lines[i] = '    ' + line

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"✓ Fixed indentation for lines {start_line+1} to {end_line+1}")
print("  Added 4 spaces (1 level) to Layer1 block")
