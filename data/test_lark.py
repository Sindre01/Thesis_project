import time
from lark import Lark
import sys
sys.path.append('../')  # Add the path to the my_packages module
from my_packages.common.rag import init_rag_data
from my_packages.evaluation.code_evaluation import get_args_from_nodes
from my_packages.utils.file_utils import read_code_file, read_dataset_to_json

# # Dynamically set the grammar file based on the nodes generated.
# node_candidates = [
#     "Add",
#     "And",
#     "Characters",
#     "Concat",
#     "Contains",
#     "Count",
#     "Difference",
#     "Div",
#     "Empty",
#     "Equal",
#     "Expression",
#     "Filter",
#     "Find",
#     "FirstItem",
#     "Flatten",
#     "Floor",
#     "For",
#     "GenerateRange",
#     "GetAt",
#     "GreaterThan",
#     "If",
#     "IfExpression",
#     "Intersection",
#     "IsEmpty",
#     "IsNumeric",
#     "LastItem",
#     "Length",
#     "LessThanOrEqual",
#     "Map",
#     "Max",
#     "Min",
#     "Modulo",
#     "Mul",
#     "Not",
#     "NotEmpty",
#     "NotEqual",
#     "Log",
#     "Pow",
#     "Reduce",
#     "Remove",
#     "Replace",
#     "Reversed",
#     "Slice",
#     "Sort",
#     "Sub",
#     "ToLower",
#     "ToUpper",
#     "Zip"
# ]

# # rag_data = init_rag_data()
# # available_args = get_args_from_nodes(node_candidates, rag_data, docs_per_node=1)
# available_args = ["result", "gen_0", "gen_1", "gen_2", "gen_3", "gen_4", "gen_5", "gen_6", "gen_7", "gen_8", "gen_9"]
# print(f"Extracted args from nodes: {available_args}")

# # Join them with a pipe to form an alternation group to use in Lark

# available_args_union = " | ".join(f'"{arg}"' for arg in available_args)
# available_nodes_union = " | ".join(f'"{node}"' for node in node_candidates)

# # Read your existing .lark file
# with open("dynamic_midio_grammar.lark", "r") as f:
#     grammar_text = f.read()


# grammar_text = grammar_text.replace("%%AVAILABLE_ARGS%%", available_args_union)
# grammar_text = grammar_text.replace("%%AVAILABLE_NODES%%", available_nodes_union)

with open("midio_grammar.lark", "r") as f:
    grammar_text = f.read()
parser = Lark(grammar_text, parser="lalr", start="start", debug=True, strict=True)

code = """
import("std", Std_k98ojb)
import("http", Http_q7o96c)

module() main {

    func(doc: "checks whether the given two integers have opposite sign or not.") opposite_signs {
        in(x: -426, y: -248, name: "x") property(Number) gen_0
        in(x: -420, y: -107, name: "y") property(Number) y_5390f5
        out(x: 159, y: -219, name: "output") property(Bool) output_3339a3

        instance(x: -208, y: -217) expression_ea12d8 root.Std_k98ojb.Math.Expression {
            expression: "(x < 0 && y > 0) || (x > 0 && y < 0)"
        }
        x_853326 -> expression_ea12d8.result
        y_5390f5 -> expression_ea12d8.gen_1
        expression_ea12d8.result -> output_3339a3
    }
    
    

    instance(x: -745, y: -368) task_id_58_77805a root.main.opposite_signs {}
} 
"""
print(code)
tree = parser.parse(code)

# # Print the raw AST
# print(tree.pretty())
print(list(parser.lex(code)))
# # time.sleep(2)
main_dataset_folder = './MBPP_Midio_50/metadata/used_external_functions'
main_dataset_folder = './MBPP_Midio_50/MBPP-Midio-50.json'
dataset = read_dataset_to_json(main_dataset_folder)

for i in range(0,50):
    # dataset[i]['code'] = dataset[i]['code'].replace("%%AVAILABLE_ARGS%%", available_args_union)
    i=i+1
    print(f"\n\nParsing code {i}...")
    code = read_code_file(i)
    print(code)
    # Parse it
    tree = parser.parse(code)

    # Print the raw AST
    # print(tree.pretty())
    print(list(parser.lex(code)))


