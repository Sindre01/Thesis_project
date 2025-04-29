import re
from typing import Any, Dict, List


def extract_function_block(code_snippet: str) -> str:
    """
    Extracts the content of the first function block found in the code snippet.
    
    """
    header_pattern = r'func\(\s*(?:doc:\s*".*?"\s*)?\)\s*\w+\s*\{'
    
    header_match = re.search(header_pattern, code_snippet, re.DOTALL)
    if not header_match:
        print("No function header found in the code snippet.")
        return ""
    
    start_index = header_match.end()
    brace_count = 1
    i = start_index
    while i < len(code_snippet) and brace_count > 0:
        if code_snippet[i] == '{':
            brace_count += 1
        elif code_snippet[i] == '}':
            brace_count -= 1
        i += 1
    
    return code_snippet[start_index:i-1]

def extract_nodes(
    code_snippet: str, 
    input_types: List[str], 
    node_types: List[str], 
    output_types: List[str],
) -> Dict[str, Any]:
    """
    Extracts nodes from a code snippet based on the defined node types.
    
    A node is assumed to be defined in the code as:
        TYPE(x: <x_value>, y: <y_value>[, name: "<node_name>"])
    where TYPE can be one of the types provided in input_types, node_types, or output_types.
    
    Returns:
        A dictionary with the following keys:
            - "input_nodes": List of input nodes.
            - "main_nodes": List of main nodes.
            - "output_nodes": List of output nodes.
            - "overall_nodes": List of all nodes.
    """
    pattern = (
        r"\b(\w+)\("                     # node type (like 'in', 'instance', etc.)
        r"x:\s*([-+]?\d+(?:\.\d+)?),"     # x coordinate (float or int)
        r"\s*y:\s*([-+]?\d+(?:\.\d+)?)"   # y coordinate (float or int)
        r"(?:,\s*name\s*(?::|=)\s*\"(.*?)\")?" # OPTIONAL: name field inside quotes
        r"\)"                            # closing parenthesis
    )


    
    matches = re.findall(pattern, code_snippet)
    if not matches:
        return {"error": "No nodes found in code snippet."}
    
    overall_nodes = []
    input_nodes = []
    main_nodes = []
    output_nodes = []
    
    for typ, x_str, y_str, name in matches:
        x = float(x_str)
        y = float(y_str)
        node_entry = {"type": typ, "x": x, "y": y}
        if name:
            node_entry["name"] = name
        overall_nodes.append(node_entry)
        
        if typ in input_types:
            input_nodes.append(node_entry)
        elif typ in output_types:
            output_nodes.append(node_entry)
        elif typ in node_types:
            main_nodes.append(node_entry)
        else:
            # If type is unknown, you may choose to treat it as a main node.
            main_nodes.append(node_entry)
    
    return {
        "input_nodes": input_nodes,
        "main_nodes": main_nodes,
        "output_nodes": output_nodes,
        "overall_nodes": overall_nodes
    }

def compute_flow_direction_score(input_nodes, main_nodes, output_nodes):
    """
    Checks to see if x_in < x_main < x_out.
    """

    flow_direction_correct_count = 0
    flow_direction_total_count = 0

    for i_node in input_nodes:
        for m_node in main_nodes:
            flow_direction_total_count += 1
            if i_node["x"] < m_node["x"]:
                flow_direction_correct_count += 1

    for m_node in main_nodes:
        for o_node in output_nodes:
            flow_direction_total_count += 1
            if m_node["x"] < o_node["x"]:
                flow_direction_correct_count += 1

    if flow_direction_total_count == 0:
        return 1.0
    
    return flow_direction_correct_count / flow_direction_total_count
def define_bounding_box(node):
    """
    Given a node with 'x', 'y', 'width', 'height',
    return the bounding box (left, right, top, bottom).
    """
    w = node.get("width", 120)
    h = node.get("height", 60)
    left   = node["x"] - w/2
    right  = node["x"] + w/2
    top    = node["y"] - h/2
    bottom = node["y"] + h/2
    return (left, right, top, bottom)

def boxes_overlap(a, b):
    """
    Return True if the bounding boxes of nodes a and b intersect.
    """
    aleft, aright, atop, abottom = define_bounding_box(a)
    bleft, bright, btop, bbottom = define_bounding_box(b)
    
    # They do NOT overlap if one is completely left/right
    # or completely above/below the other
    if aright < bleft or aleft > bright:
        return False
    if abottom < btop or atop > bbottom:
        return False
    return True

def compute_overlap_score(nodes):
    """
    A node is only considered correct if it does NOT overlap with ANY other node.
    Computes the fraction of nodes that are fully non-overlapping.
    """
    if not nodes:
        return 1.0

    non_overlapping_nodes = 0

    for i, node in enumerate(nodes):
        has_overlap = False
        for j, other_node in enumerate(nodes):
            if i != j and boxes_overlap(node, other_node):
                has_overlap = True
                break
        if not has_overlap:
            non_overlapping_nodes += 1

    return non_overlapping_nodes / len(nodes)


def evaluate_visual_flow(
    code_snippet: str, 
    input_types: List[dict] = None, 
    node_types: List[dict] = None, 
    output_types: List[dict] = None
) -> Dict[str, Any]:
    """
    Evaluates the visual flow of a code snippet based on the defined node types.
    
    It extracts nodes (only the ones inside the first function block found) and calculates:
      - Flow direction (inputs to left, main nodes in the middle, outputs to right)
      - Overlap (nodes should not overlap in bounding boxes)

    Returns a dictionary with the extracted nodes and computed metrics.
    """
    # use default sizes
    if input_types is None:
        input_types = [
            {"name": "in", "height": 50, "width": 100}
        ]
    if node_types is None:
        node_types = [
            {"name": "instance", "height": 100, "width": 200},
            {"name": "data_instance", "height": 100, "width": 200},
            {"name": "setter", "height": 100, "width": 200},
            {"name": "getter", "height": 100, "width": 200},
            {"name": "waypoint", "height": 100, "width": 200}
        ]
    if output_types is None:
        output_types = [
            {"name": "out", "height": 50, "width": 100}
        ]
        
    # Extract the first function block
    function_block = extract_function_block(code_snippet)
    if not function_block:
        print("No function block found in the code snippet.")
        return 0.0
    # print("Function block:", function_block)

    input_type_names  = [t["name"] for t in input_types]
    node_type_names   = [t["name"] for t in node_types]
    output_type_names = [t["name"] for t in output_types]

    # Extract actual nodes with x,y from code
    print(function_block)
    nodes = extract_nodes(function_block, input_type_names, node_type_names, output_type_names)
    # print("Nodes:", nodes)

    input_nodes = nodes.get("input_nodes", [])
    main_nodes  = nodes.get("main_nodes", [])
    output_nodes = nodes.get("output_nodes", [])
    overall_nodes = nodes.get("overall_nodes", [])
    if not overall_nodes:
        print("No nodes found in the function block.")
        return 0.0


    size_map = {}
    for group in (input_types, node_types, output_types):
        for tdef in group:
            # e.g. {"name": "in", "height": 60, "width": 120}
            size_map[tdef["name"]] = (tdef["height"], tdef["width"])
    
    for node in overall_nodes:
        h, w = size_map.get(node["type"], (60,120))
        node["height"] = h
        node["width"]  = w

    # FLOW DIRECTION CHECK
    flow_direction_score = compute_flow_direction_score(input_nodes, main_nodes, output_nodes)
    
    # OVERLAPPING NODES CHECK
    overlap_score = compute_overlap_score(overall_nodes)

    overall_score = 0.3 * flow_direction_score + 0.7 * overlap_score # Overlap_score is twice as important

    result = {
        "input_nodes": input_nodes,
        "main_nodes": main_nodes,
        "output_nodes": output_nodes,
        "flow_direction_score": flow_direction_score,
        "overlap_score": overlap_score,
        "overall_score": overall_score
    }
    print(result)
    print(f"Flow Direction Score: {flow_direction_score:.2f}")
    print(f"Overlap Score: {overlap_score:.2f}")
    print(f"Overall Score: {overall_score:.2f}")
    return result["overall_score"]




