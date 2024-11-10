# Dataset Format Explanation

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

- **task_id**: An integer uniquely identifying the task within the dataset.

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

## Purpose of the Dataset

This dataset is designed to create the textual representation of tasks in the visual and flow-based programming language Midio. It provides:

- **Comprehensive Task Descriptions**: Detailed prompts and specifications guide the implementation of each task.
- **Metadata for Implementation**: Information about library functions, visual node types, and textual instance types assists in constructing the flow.
- **Testing Information**: The `testing` object contains all necessary details to verify the correctness of the implementation.
- **Flexibility**: The format supports various programming paradigms and can be extended with additional fields as needed.

## Usage Notes

- **Task Identification**: Each task is uniquely identified by `task_id` and `MBPP_task_id`, allowing for easy reference and organization.

- **Implementation Details**: The `prompts` and `specification` fields provide step-by-step guidance on how to implement the task, including expected behaviors and constraints.

- **Testing Components**: The `testing` object specifies the functions and visual nodes required for testing. This ensures that implementations can be validated effectively.

- **Visual Programming Support**: Fields like `visual_node_types` and `testing.visual_node_types` cater to visual or flow-based programming environments, outlining the nodes and components to be used.

- **Extensibility**: The dataset format is adaptable, allowing for the inclusion of additional fields or modifications to suit different programming needs.

---

By overriding the README with this updated information, it now reflects the new format you provided, ensuring clarity and consistency for users of the dataset.