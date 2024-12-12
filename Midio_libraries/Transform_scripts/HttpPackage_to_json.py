import re
import json

def extract_functions(filename):
    functions = []
    module_stack = []
    brace_stack = []
    in_function_body = False
    in_event_body = False
    current_body = []
    internal_brace_counter = 0
    types_stack = []

    # Regular expressions to match module, function, event, and type definitions
    module_pattern = re.compile(r'^module(?:\s*\(.*?\))?\s+(\w+)\s*\{')
    func_pattern = re.compile(r'^\s*(?:extern\s+)?func(?:\s*\(.*?\))?\s+(\w+)\s*\{')
    func_with_doc_pattern = re.compile(r'^\s*extern\s+func\s*\(\s*doc\s*:\s*"([^"]*)"\s*\)\s+(\w+)\s*\{')
    event_with_doc_pattern = re.compile(r'^\s*extern\s+event\s*\(\s*doc\s*:\s*"([^"]*)"\s*\)\s+(\w+)\s*\{')
    event_pattern = re.compile(r'^\s*(?:extern\s+)?event(?:\s*\(.*?\))?\s+(\w+)\s*\{')
    type_pattern = re.compile(r'^type\s+(\w+)\s+(\w+)')

    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, 1):
            original_line = line  # Keep the original line for debugging
            line = line.strip()

            # If currently inside a function or event body, capture the body
            if in_function_body or in_event_body:
                current_body.append(original_line)
                opening_braces = line.count('{')
                closing_braces = line.count('}')
                internal_brace_counter += opening_braces - closing_braces

                if internal_brace_counter <= 0:
                    # Function or event body ends
                    in_function_body = False
                    in_event_body = False
                    body_str = '\n'.join(current_body)
                    if functions:
                        functions[-1]['body'] = body_str.strip()
                    current_body = []
                    # Pop the 'function' or 'event' scope from brace_stack
                    if brace_stack and brace_stack[-1] in ['function', 'event']:
                        brace_stack.pop()
                continue

            # Check for module start
            module_match = module_pattern.match(line)
            if module_match:
                module_name = module_match.group(1)
                module_stack.append(module_name)
                brace_stack.append('module')
                types_stack.append([])  # Initialize types list for the new module
                print(
                    f"Line {line_number}: Entered module '{module_name}'. Current module stack: {module_stack}")
                continue

            # Check for type definition
            type_match = type_pattern.match(line)
            if type_match and module_stack:
                type_name = type_match.group(1)
                base_type = type_match.group(2)
                type_definition = f"type {type_name} {base_type}"
                types_stack[-1].append(type_definition)
                print(
                    f"Line {line_number}: Defined type '{type_name}' with base type '{base_type}' in module '{module_stack[-1]}'.")
                continue

            # Check for event definition with doc string
            event_with_doc_match = event_with_doc_pattern.match(line)
            if event_with_doc_match:
                doc_string = event_with_doc_match.group(1)
                event_name = event_with_doc_match.group(2)
                brace_stack.append('event')  # Track event scope to handle closing braces
                # Build the full module path by joining all module names in the stack
                current_module = '.'.join(module_stack)
                # Construct the full event name
                full_event_name = f"{current_module}.{event_name}" if current_module else event_name

                # Aggregate types from all parent modules
                aggregated_types = "\n".join([t for types in types_stack for t in types])

                functions.append({
                    'type': 'event',
                    'function_name': full_event_name,
                    'module_path': current_module,
                    'doc': doc_string,
                    'body': "",
                    'types': aggregated_types
                })
                in_event_body = True
                internal_brace_counter = 1  # Initialize brace counter with the opening brace of the event
                current_body = [original_line]
                print(
                    f"Line {line_number}: Found event '{full_event_name}' with doc: '{doc_string}'.")
                continue

            # Check for event definition without doc string
            event_match = event_pattern.match(line)
            if event_match:
                event_name = event_match.group(1)
                brace_stack.append('event')  # Track event scope to handle closing braces
                # Build the full module path by joining all module names in the stack
                current_module = '.'.join(module_stack)
                # Construct the full event name
                full_event_name = f"{current_module}.{event_name}" if current_module else event_name

                # Aggregate types from all parent modules
                aggregated_types = "\n".join([t for types in types_stack for t in types])

                functions.append({
                    'type': 'event',
                    'function_name': full_event_name,
                    'module_path': current_module,
                    'doc': "",
                    'body': "",
                    'types': aggregated_types
                })
                in_event_body = True
                internal_brace_counter = 1  # Initialize brace counter with the opening brace of the event
                current_body = [original_line]
                print(
                    f"Line {line_number}: Found event '{full_event_name}'.")
                continue

            # Check for function definition with doc string
            func_with_doc_match = func_with_doc_pattern.match(line)
            if func_with_doc_match:
                doc_string = func_with_doc_match.group(1)
                func_name = func_with_doc_match.group(2)
                # Build the full module path by joining all module names in the stack
                current_module = '.'.join(module_stack)
                # Construct the full function name
                full_function_name = f"{current_module}.{func_name}" if current_module else func_name

                # Aggregate types from all parent modules
                aggregated_types = "\n".join([t for types in types_stack for t in types])

                functions.append({
                    'type': 'function',
                    'function_name': full_function_name,
                    'module_path': current_module,
                    'doc': doc_string,
                    'body': "",
                    'types': aggregated_types
                })
                brace_stack.append('function')
                in_function_body = True
                internal_brace_counter = 1  # Initialize brace counter with the opening brace of the function
                current_body = [original_line]
                print(
                    f"Line {line_number}: Defined function '{full_function_name}' with doc: '{doc_string}'.")
                continue

            # Check for function definition without doc string
            func_match = func_pattern.match(line)
            if func_match:
                func_name = func_match.group(1)
                # Build the full module path by joining all module names in the stack
                current_module = '.'.join(module_stack)
                # Construct the full function name
                full_function_name = f"{current_module}.{func_name}" if current_module else func_name

                # Aggregate types from all parent modules
                aggregated_types = "\n".join([t for types in types_stack for t in types])

                functions.append({
                    'type': 'function',
                    'function_name': full_function_name,
                    'module_path': current_module,
                    'doc': "",
                    'body': "",
                    'types': aggregated_types
                })
                brace_stack.append('function')
                in_function_body = True
                internal_brace_counter = 1  # Initialize brace counter with the opening brace of the function
                current_body = [original_line]
                print(
                    f"Line {line_number}: Defined function '{full_function_name}' without doc.")
                continue

            # Check for closing brace
            if line == '}':
                if brace_stack:
                    closed_scope = brace_stack.pop()
                    if closed_scope == 'module':
                        if module_stack:
                            popped_module = module_stack.pop()
                            types_stack.pop()  # Remove types of the exited module
                            print(
                                f"Line {line_number}: Exited module '{popped_module}'. Current module stack: {module_stack}")
                    elif closed_scope == 'event':
                        print(
                            f"Line {line_number}: Exited event scope.")
                    elif closed_scope == 'function':
                        print(
                            f"Line {line_number}: Exited function scope.")
                else:
                    print(
                        f"Warning: Unmatched closing brace at line {line_number}")
                continue

    # Check for any unmatched opening braces
    if brace_stack:
        print("Warning: There are unmatched opening braces in the file.")

    return functions

# Path to the provided file
input_file = 'HttpPackage'  # Replace with the correct path to your HttpPackage file

# Extract functions and their module paths
function_list = extract_functions(input_file)

# Save the result to a JSON file
output_file = 'HttpPackage_dataset.json'
with open(output_file, 'w') as json_file:
    json.dump(function_list, json_file, indent=4)

print(f"Extracted {len(function_list)} functions and saved to {output_file}")