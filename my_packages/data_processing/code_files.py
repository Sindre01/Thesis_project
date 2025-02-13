import re

from my_packages.utils.file_utils import get_test_module_from_file, write_code_file, read_test_code_file, write_code_file
def extract_all_code_and_write_to_file():
    """ Extracts code wthout tests module from the files in includes_files folder and writes to files only_files folder"""
    for i in range(50):
        code = read_test_code_file(i+1)
        print(f"Code {i+1}: {code}")
        test_module = get_test_module_from_file(i+1)
        removed_module= code.replace(test_module, "")
        print(f"Removed module: {removed_module}")
        write_code_file(i+1, removed_module)

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
