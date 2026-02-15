#!/usr/bin/env python3
"""
Validate documentation code snippets.

This script extracts Python code blocks from markdown documentation files
and validates their syntax. It can also optionally test imports and basic
execution (for simple examples).

Usage:
    python scripts/validate_docs.py [--test-imports] [--execute] [files...]

If no files are specified, checks all .md files in docs/ directory.
"""

import argparse
import ast
import doctest
import io
import os
import re
import sys
import traceback
import contextlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Patterns to skip validation for
SKIP_PATTERNS = [
    r"^\s*#\s*TODO",
    r"^\s*#\s*FIXME",
    r"^\s*#\s*Example output",
    r"^\s*#\s*Output:",
    r"^\s*#\s*>>>",  # Doctest examples
    r"^\s*\.\.\.",  # Doctest continuation
]

# Imports that may not be available in validation environment
# but are valid in actual usage
ALLOWED_MISSING_IMPORTS = {
    "cellconstructor",
    "cellconstructor.Phonons",
    "cellconstructor.symmetries",
    "cellconstructor.Settings",
    "sscha",
    "sscha.Ensemble",
    "sscha.Parallel",
    "tdscha",
    "tdscha.DynamicalLanczos",
    "tdscha.StaticHessian",
    "tdscha.Tools",
    "tdscha.Perturbations",
    "tdscha.Parallel",
    "sscha_HP_odd",
    "julia",
    "julia.Main",
    "mpi4py",
    "spglib",
}


def extract_python_blocks(filepath: Path) -> List[Tuple[int, str]]:
    """Extract Python code blocks from markdown file.
    
    Returns list of (line_number, code) tuples.
    """
    blocks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_code_block = False
    code_block_lines = []
    block_start_line = 0
    language = ""
    
    for i, line in enumerate(lines, 1):
        # Check for code block start
        match = re.match(r'^```(\w*)', line.strip())
        if match and not in_code_block:
            in_code_block = True
            language = match.group(1).lower()
            block_start_line = i
            code_block_lines = []
        elif line.strip() == '```' and in_code_block:
            in_code_block = False
            if language == 'python' and code_block_lines:
                blocks.append((block_start_line, '\n'.join(code_block_lines)))
            language = ""
            code_block_lines = []
        elif in_code_block:
            code_block_lines.append(line.rstrip('\n'))
    
    return blocks


def validate_syntax(code: str, line_offset: int = 0) -> List[str]:
    """Validate Python syntax, return list of errors."""
    errors = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        # Adjust line number for markdown context
        adjusted_line = e.lineno + line_offset if e.lineno else line_offset
        errors.append(f"  Line {adjusted_line}: {e.msg}")
    
    return errors


def check_imports(code: str) -> List[str]:
    """Check for potentially problematic imports."""
    warnings = []
    
    # Simple import detection
    import_pattern = r'^\s*(import|from)\s+(\w+)'
    for line in code.split('\n'):
        match = re.match(import_pattern, line)
        if match:
            module = match.group(2)
            if module in ALLOWED_MISSING_IMPORTS:
                warnings.append(f"  May require {module} to be installed")
    
    return warnings


def should_skip(code: str) -> bool:
    """Determine if code block should be skipped."""
    lines = code.split('\n')
    for line in lines:
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, line):
                return True
    return False


def extract_doctest_blocks(filepath: Path) -> List[Tuple[int, List[doctest.Example]]]:
    """Extract doctest blocks from markdown file using doctest parser.
    
    Returns list of (line_number, list_of_examples) tuples.
    Each block contains one or more contiguous doctest examples.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    parser = doctest.DocTestParser()
    try:
        # Parse the entire file text
        test = parser.get_doctest(text, {}, filepath.name, str(filepath), 0)
    except Exception:
        # If parsing fails, fall back to simple extraction
        return _extract_doctest_blocks_simple(filepath)
    
    blocks = []
    current_block = []
    current_start = None
    last_line = -10
    
    for example in test.examples:
        # Get the line number (1-indexed)
        lineno = example.lineno
        
        # Check if this example is contiguous with previous one
        # (within 5 lines and same paragraph)
        if current_block and lineno <= last_line + 5:
            # Continue current block
            current_block.append(example)
            last_line = max(last_line, lineno + example.source.count('\n'))
        else:
            # Start new block
            if current_block:
                # Save previous block
                blocks.append((current_start, current_block))
            
            current_block = [example]
            current_start = lineno
            last_line = lineno + example.source.count('\n')
    
    # Add last block
    if current_block:
        blocks.append((current_start, current_block))
    
    return blocks


def _parse_doctest_text(text: str) -> List[doctest.Example]:
    """Parse doctest text into a list of Example objects."""
    parser = doctest.DocTestParser()
    try:
        test = parser.get_doctest(text, {}, '<string>', '<string>', 0)
        return test.examples
    except Exception:
        # If parsing fails, return empty list
        return []

def _extract_doctest_blocks_simple(filepath: Path) -> List[Tuple[int, List[doctest.Example]]]:
    """Simple fallback extraction for when doctest parser fails."""
    blocks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_doctest = False
    doctest_lines = []
    block_start_line = 0
    
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip('\n')
        
        # Check for doctest prompt
        if stripped.lstrip().startswith('>>>'):
            if not in_doctest:
                # Start new doctest block
                in_doctest = True
                block_start_line = i
                doctest_lines = [stripped]
            else:
                # Continue existing block
                doctest_lines.append(stripped)
        elif in_doctest and stripped.lstrip().startswith('...'):
            # Continuation line
            doctest_lines.append(stripped)
        elif in_doctest:
            # Could be expected output or blank line
            if stripped == '':
                # Blank line might separate examples
                # Check if next line has >>> (we can't see ahead)
                # For simplicity, treat blank line as end of block
                in_doctest = False
                if doctest_lines:
                    text_block = '\n'.join(doctest_lines)
                    examples = _parse_doctest_text(text_block)
                    if examples:
                        blocks.append((block_start_line, examples))
                doctest_lines = []
            else:
                # Expected output, keep in block
                doctest_lines.append(stripped)
    
    # Handle doctest block at end of file
    if in_doctest and doctest_lines:
        text_block = '\n'.join(doctest_lines)
        examples = _parse_doctest_text(text_block)
        if examples:
            blocks.append((block_start_line, examples))
    
    return blocks


def _examples_to_text(examples: list) -> str:
    """Convert list of doctest.Example objects to text."""
    lines = []
    for ex in examples:
        lines.append(ex.source.rstrip())
        if ex.want:
            lines.append(ex.want.rstrip())
    return '\n'.join(lines)


def create_test_globals() -> Dict[str, Any]:
    """Create globals dictionary for doctest execution.
    
    Includes tdscha modules and test utilities.
    """
    globals_dict = {}
    
    # Add standard imports
    import numpy as np
    globals_dict['np'] = np
    globals_dict['numpy'] = np
    
    # Try to import tdscha normally (if installed)
    tdscha_module = None
    try:
        import tdscha
        tdscha_module = tdscha
    except ImportError:
        # Fallback: create tdscha module from Modules directory
        import sys
        import os
        import types
        # Project root is parent directory of this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        modules_dir = os.path.join(project_root, 'Modules')
        if os.path.exists(os.path.join(modules_dir, '__init__.py')):
            # Insert project root to sys.path (so Modules can be imported)
            sys.path.insert(0, project_root)
            try:
                # Import Modules and its submodules
                import Modules
                # Import submodules (they may import dependencies that could fail)
                import Modules.DynamicalLanczos
                import Modules.StaticHessian
                import Modules.Tools
                import Modules.Perturbations
                import Modules.Parallel
                import Modules.testing
                # Create a new module to act as tdscha
                tdscha_module = types.ModuleType('tdscha')
                # Assign submodules as attributes
                tdscha_module.DynamicalLanczos = Modules.DynamicalLanczos
                tdscha_module.StaticHessian = Modules.StaticHessian
                tdscha_module.Tools = Modules.Tools
                tdscha_module.Perturbations = Modules.Perturbations
                tdscha_module.Parallel = Modules.Parallel
                tdscha_module.testing = Modules.testing
                # Install in sys.modules for import statements
                sys.modules['tdscha'] = tdscha_module
                sys.modules['tdscha.DynamicalLanczos'] = Modules.DynamicalLanczos
                sys.modules['tdscha.StaticHessian'] = Modules.StaticHessian
                sys.modules['tdscha.Tools'] = Modules.Tools
                sys.modules['tdscha.Perturbations'] = Modules.Perturbations
                sys.modules['tdscha.Parallel'] = Modules.Parallel
                sys.modules['tdscha.testing'] = Modules.testing
                sys.modules['tdscha.testing.test_data'] = Modules.testing.test_data
            except ImportError as e:
                tdscha_module = None
            finally:
                sys.path.pop(0)
    
    if tdscha_module is not None:
        globals_dict['tdscha'] = tdscha_module
        
        # Try to import submodules and assign them to tdscha_module attributes
        submodules = [
            'DynamicalLanczos',
            'StaticHessian', 
            'Tools',
            'Perturbations',
            'Parallel',
            'testing'
        ]
        
        for submod_name in submodules:
            try:
                full_name = f'tdscha.{submod_name}'
                # Import the submodule
                imported = __import__(full_name, fromlist=[''])
                # Assign as attribute to tdscha_module if not already present
                if not hasattr(tdscha_module, submod_name):
                    setattr(tdscha_module, submod_name, imported)
                # Add to globals_dict for direct access
                globals_dict[full_name] = imported
            except ImportError:
                # Submodule not available, skip
                pass
    
    # Try to add cellconstructor and sscha (these are external dependencies)
    try:
        import cellconstructor as CC
        globals_dict['CC'] = CC
        globals_dict['cellconstructor'] = CC
        globals_dict['cellconstructor.Phonons'] = CC.Phonons
    except ImportError:
        pass
    
    try:
        import sscha
        globals_dict['sscha'] = sscha
        globals_dict['sscha.Ensemble'] = sscha.Ensemble
    except ImportError:
        pass
    
    # Try to add test_data utilities
    if tdscha_module is not None:
        try:
            # Import from tdscha.testing.test_data (now available via sys.modules)
            from tdscha.testing.test_data import (
                load_test_ensemble, create_test_lanczos, get_test_mode_frequencies
            )
            globals_dict['load_test_ensemble'] = load_test_ensemble
            globals_dict['create_test_lanczos'] = create_test_lanczos
            globals_dict['get_test_mode_frequencies'] = get_test_mode_frequencies
        except ImportError:
            # Fallback: try to import from source directory
            import sys
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            modules_dir = os.path.join(project_root, 'Modules')
            if os.path.exists(os.path.join(modules_dir, '__init__.py')):
                sys.path.insert(0, project_root)
                try:
                    # Import Modules.testing.test_data
                    import Modules.testing.test_data
                    from Modules.testing.test_data import (
                        load_test_ensemble, create_test_lanczos, get_test_mode_frequencies
                    )
                    globals_dict['load_test_ensemble'] = load_test_ensemble
                    globals_dict['create_test_lanczos'] = create_test_lanczos
                    globals_dict['get_test_mode_frequencies'] = get_test_mode_frequencies
                    # Also make available via tdscha.testing for import statements
                    sys.modules['tdscha.testing'] = Modules.testing
                    sys.modules['tdscha.testing.test_data'] = Modules.testing.test_data
                except ImportError:
                    pass
                finally:
                    sys.path.pop(0)
    
    return globals_dict


def run_doctest_block(examples: List[doctest.Example], globals_dict: Dict[str, Any], 
                      verbose: bool = False) -> doctest.TestResults:
    """Run a single doctest block."""
    # Create a doctest runner with ELLIPSIS flag to ignore variable output
    runner = doctest.DocTestRunner(verbose=verbose, 
                                   optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    
    # Create a test from the examples
    test = doctest.DocTest(
        examples=examples,
        globs=globals_dict,
        name='<doctest>',
        filename='<string>',
        lineno=0,
        docstring=''
    )
    
    # Run the test
    runner.run(test)
    return runner.summarize()


def validate_file(filepath: Path, test_imports: bool = False, 
                  run_doctests: bool = False, verbose: bool = False) -> bool:
    """Validate Python code blocks and optionally run doctests."""
    all_valid = True
    
    # 1. Validate regular Python code blocks
    print(f"Validating {filepath}...")
    blocks = extract_python_blocks(filepath)
    
    if blocks:
        for block_num, (line_num, code) in enumerate(blocks, 1):
            if should_skip(code):
                print(f"  Code block {block_num} (line {line_num}): Skipped")
                continue
            
            print(f"  Code block {block_num} (line {line_num}): ", end="")
            
            # Validate syntax
            errors = validate_syntax(code, line_num - 1)
            
            if errors:
                print("FAILED")
                all_valid = False
                for error in errors:
                    print(f"    {error}")
            else:
                print("Syntax OK")
            
            # Check imports if requested
            if test_imports:
                warnings = check_imports(code)
                if warnings:
                    print(f"    Import warnings:")
                    for warning in warnings:
                        print(f"    {warning}")
    else:
        print(f"  No Python code blocks found")
    
    # 2. Run doctests if requested
    if run_doctests:
        print(f"  Running doctests...")
        doctest_blocks = extract_doctest_blocks(filepath)
        
        if not doctest_blocks:
            print(f"  No doctest blocks found")
        else:
            # Create test globals for doctests
            globals_dict = create_test_globals()
            
            for block_num, (line_num, examples) in enumerate(doctest_blocks, 1):
                print(f"  Doctest block {block_num} (line {line_num}): ", end="")
                
                try:
                    # Run the doctest block
                    failure_count, test_count = run_doctest_block(
                        examples, globals_dict, verbose
                    )
                    
                    if failure_count == 0:
                        print(f"PASSED ({test_count} examples)")
                    else:
                        print(f"FAILED ({failure_count}/{test_count} failures)")
                        all_valid = False
                        
                        # For verbose output, show the failing doctest
                        if verbose:
                            print(f"    Doctest examples:")
                            for i, ex in enumerate(examples, 1):
                                print(f"      Example {i}:")
                                print(f"        Source: {ex.source.rstrip()}")
                                if ex.want:
                                    print(f"        Expected: {ex.want.rstrip()}")
                            
                except Exception as e:
                    print(f"ERROR (exception: {e})")
                    all_valid = False
                    if verbose:
                        import traceback
                        traceback.print_exc()
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Validate documentation code snippets")
    parser.add_argument("files", nargs="*", help="Markdown files to validate")
    parser.add_argument("--test-imports", action="store_true", 
                       help="Check for potentially missing imports")
    parser.add_argument("--execute", action="store_true",
                       help="Execute code blocks (dangerous, not implemented)")
    parser.add_argument("--doctest", action="store_true",
                       help="Run doctests (>>> examples) using test data")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show more detailed output")
    
    args = parser.parse_args()
    
    # Determine files to validate
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        docs_dir = Path("docs")
        files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("**/*.md"))
    
    # Validate each file
    all_valid = True
    for filepath in files:
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        valid = validate_file(filepath, args.test_imports, args.doctest, args.verbose)
        if not valid:
            all_valid = False
    
    if all_valid:
        print("\nAll code blocks validated successfully!")
        return 0
    else:
        print("\nSome code blocks failed validation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())