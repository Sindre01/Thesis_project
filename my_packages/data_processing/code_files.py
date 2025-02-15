import re

def extract_func_signature(code_str: str) -> str:
    """
    Extracts the top-level function signature block (header, and only the 'in(' or 'out(' lines
    at the top-level, ignoring nested blocks) from a given code string.
    
    For example, given:
    
    func(doc: "Finds the product of first even and odd number of a given list.") mul_even_odd {
        in(x: -277, y: 166, name: "list") property(List) list_09fcba

        out(x: 800, y: 145, name: "output") property(Number) output_edc2e3

        instance(x: 532, y: 147) mul_7e1ce0 root.Std_k98ojb.Math.Mul {}
        ... (nested code) ...
    }
    
    It returns only:
    
    func(doc: "Finds the product of first even and odd number of a given list.") mul_even_odd {
        in(x: -277, y: 166, name: "list") property(List) list_09fcba
        out(x: 800, y: 145, name: "output") property(Number) output_edc2e3
    }
    """
    # Use a regex to extract a function block starting at 'func(' and ending at the corresponding closing brace.
    # This pattern assumes the block is balanced.
    pattern = re.compile(r'(func\(.*?\{.*?\})', re.DOTALL)
    match = pattern.search(code_str)
    if not match:
        print("No function block found.")
        return "Not found"
    
    block = match.group(1)
    lines = block.splitlines()
    if not lines:
        return ""
    
    # We'll build the output while tracking the nesting level.
    # Assume the header (first line) is always at level 0.
    result_lines = []
    nesting = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        # Count braces in this line (assumes braces don't occur inside strings)
        opens = line.count("{")
        closes = line.count("}")
        
        # Before processing the line, if it's the header, include it.
        if idx == 0:
            result_lines.append(line)
            # Increase nesting level if header contains '{'
            nesting += opens - closes
            continue

        # Update nesting AFTER checking if this line is a candidate.
        # We want to include only lines that are at the top level (nesting == 1).
        # That is, after the header, the first level of content.
        if nesting == 1 and (line.lstrip().startswith("in(") or line.lstrip().startswith("out(")):
            result_lines.append(line)
        nesting += opens - closes

    # Optionally, append the closing brace for the top-level function.
    # We assume that when the block ends, the nesting should be 0.
    # Here, we add a closing brace if the last non-empty line isn't one.
    if result_lines and result_lines[-1].strip() != "}":
        result_lines.append("}")
    
    return "\n".join(result_lines)


def format_func_string(code: str) -> str:
    """Convert string to an inline string with escaped quotes and tabs after newlines."""
    func_string = extract_func_signature(code)
    # Escape double quotes by replacing " with \"
    func_string = func_string.replace('"', '\"')
    # Replace newlines followed by any whitespace with a newline and a tab.
    func_string = re.sub(r'\n\s+', '\n\t', func_string)
    return func_string

def find_matching_brace(text: str, start_index: int) -> int:
    """
    Given a string and the index of an opening brace '{', return the index of its
    corresponding closing brace '}'. Returns -1 if no matching brace is found.
    """
    if text[start_index] != '{':
        raise ValueError(f"Expected '{{' at position {start_index}")
    
    count = 0
    for i in range(start_index, len(text)):
        if text[i] == '{':
            count += 1
        elif text[i] == '}':
            count -= 1
            if count == 0:
                return i
    return -1

def extract_tests_module(content: str) -> str:
    """
    Extracts the tests module block directly from the content using an iterative
    approach for matching braces.
    
    Expected content example:
    
        import("std", Std_k98ojb)
        import("http", Http_q7o96c)
    
        module() main { 
            func(doc: "Finds the last position of an element in a sorted array.") last {
                in(x: -231, y: -29, name: "list") property(List) list_2bbadf
            }
    
            module(doc: "Contains three different tests for the 'last' function node") tests {
                instance(x: -359, y: 161.5) test_e0e516 root.Std_k98ojb.Testing.Test {
                    name: "Test last"
                }
            }
    
            instance(x: -203, y: -53.5) last_9181e6 root.main.last {}
        }
    
    Returns:
        The entire tests module block (including header and balanced braces) as a string,
        or an empty string if not found.
    """
    # Use a simple pattern to locate the tests module header.
    header_pattern = re.compile(r'module\s*\([^)]*\)\s*tests\s*\{', re.MULTILINE)
    header_match = header_pattern.search(content)
    if not header_match:
        print("Tests module header not found")
        return ""
    
    # Get the position of the opening brace for the tests block.
    block_start = header_match.end() - 1  # This is the '{'
    block_end = find_matching_brace(content, block_start)
    if block_end == -1:
        print("Matching closing brace for tests module not found")
        return ""
    
    # Return the substring from the header start to the matching closing brace.
    return content[header_match.start(): block_end + 1]
