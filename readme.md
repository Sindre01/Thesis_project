# Experiments

## Few-shot
see path: ..

## RAG and full Midio documentation in-context (context)
see path: ..

## SynCode
see path: ..

## Self-Debugging (Refinement)
see path: ..

## Visual_flow_metric
see path: ..


# Dataset: MBPP-Midio-50

This dataset consists of JSON objects, each representing a programming task designed for the Midio visual and flow-based programming language. Each object contains detailed information about the task, including prompts, specifications, identifiers, and metadata related to library functions and visual components.

## Sample JSON Object

```json
{
    "prompts": [
        "Create a flow to multiply two numbers.",
        "The flow should create a 'Math Expression' node with the expression 'x * y'.",
        "A test should be created as there are no fixed numbers provided, and the output is not known. The user-defined function should then be tested in the main module with 'Testing Test' and 'Testing AssertEqual' nodes." 
    ],
    "task_id": 5,
    "specification": {
        "function_signature": "",
        "preconditions": "- There are no preconditions, the method will always work.",
        "postconditions": "- The result should be the product of the two input integers"
    },
    "MBPP_task_id": 127,
    "library_functions": [
        "root.std.Math.Expression"
    ],
    "visual_node_types": [
        "Function",
        "Data Object"
    ],
    "textual_instance_types": [
        "instance",
        "data_instance"
    ],
    "testing": {
        "library_functions": [
            "root.std.Testing.Test",
            "root.std.Testing.AssertEqual"
        ],
        "visual_node_types": [
            "Event",
            "Function",
            "Data Object"
        ],
        "textual_instance_types": [
            "instance",
            "data_instance"
        ]
        
    }
}
```

## Field Descriptions

- **prompts**: An array of strings providing instructions and descriptions for the task.
    - `prompts[0]`: The main task description.
    - `prompts[1]`: Detailed implementation instructions.
    - `prompts[2]`: Information about testing the implementation.

- **task_id**: An integer uniquely identifying the task within the dataset. Every 

- **specification**: An object containing detailed specifications of the task.
    - **function_signature**: A string defining the function or method signature.
    - **preconditions**: A string detailing any conditions that must be met before execution.
    - **postconditions**: A string describing the expected outcomes after execution.

- **MBPP_task_id**: An integer uniquely identifying the task within the MBPP dataset. -1 means that the task was not taken from MBPP

- **library_functions**: An array of strings listing the library functions relevant to the task.

- **visual_node_types**: An array of strings representing the types of visual nodes used in the task.

- **textual_instance_types**: An array of strings indicating the types of textual instances or elements involved in the task.

- **testing**: An object containing information about testing the task.
    - **library_functions**: An array of strings specifying the library functions or modules used for testing the solution.
    - **visual_node_types**: An array of strings representing the types of visual nodes used specifically for testing the task.
    - **textual_instance_types**: An array of strings indicating the types of textual instances or elements involved in testing.

