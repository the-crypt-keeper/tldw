# import_fixer.py - Fix incorrect imports in RAG module
"""
Script to fix incorrect imports from tldw_chatbook to tldw_Server_API
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def find_files_with_incorrect_imports(directory: str) -> List[Path]:
    """Find all Python files with incorrect imports."""
    files_to_fix = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'from tldw_chatbook' in content or 'import tldw_chatbook' in content:
                        files_to_fix.append(filepath)
    
    return files_to_fix

def fix_imports_in_file(filepath: Path) -> List[Tuple[str, str]]:
    """Fix imports in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # Common import mappings
    replacements = [
        # Direct module replacements
        (r'from tldw_chatbook\.', 'from tldw_Server_API.app.core.'),
        (r'import tldw_chatbook\.', 'import tldw_Server_API.app.core.'),
        
        # Specific module mappings
        (r'tldw_chatbook\.RAG_Search', 'tldw_Server_API.app.core.RAG.RAG_Search'),
        (r'tldw_chatbook\.LLM_Calls', 'tldw_Server_API.app.core.LLM_Calls'),
        (r'tldw_chatbook\.Metrics', 'tldw_Server_API.app.core.Metrics'),
        (r'tldw_chatbook\.config', 'tldw_Server_API.app.core.Config'),
        (r'tldw_chatbook\.DB_Management', 'tldw_Server_API.app.core.DB_Management'),
        (r'tldw_chatbook\.utils', 'tldw_Server_API.app.core.Utils'),
    ]
    
    for old_pattern, new_pattern in replacements:
        if re.search(old_pattern, content):
            old_matches = re.findall(old_pattern + r'[^\s]+', content)
            content = re.sub(old_pattern, new_pattern, content)
            for match in old_matches:
                changes.append((match, match.replace('tldw_chatbook', 'tldw_Server_API.app.core')))
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return changes

def main():
    """Run the import fixer."""
    rag_directory = Path(__file__).parent / 'RAG_Search'
    
    print(f"Scanning directory: {rag_directory}")
    
    files_to_fix = find_files_with_incorrect_imports(str(rag_directory))
    
    if not files_to_fix:
        print("No files with incorrect imports found.")
        return
    
    print(f"\nFound {len(files_to_fix)} files with incorrect imports:")
    for filepath in files_to_fix:
        print(f"  - {filepath.relative_to(rag_directory.parent)}")
    
    print("\nFixing imports...")
    
    total_changes = []
    for filepath in files_to_fix:
        changes = fix_imports_in_file(filepath)
        if changes:
            print(f"\n{filepath.name}:")
            for old, new in changes:
                print(f"  {old} -> {new}")
            total_changes.extend(changes)
    
    print(f"\nTotal changes made: {len(total_changes)}")

if __name__ == "__main__":
    main()