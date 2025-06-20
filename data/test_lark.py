import time
import sys
sys.path.append('../')  # Add the path to the my_packages module
from lark import Lark
from my_packages.common.rag import init_rag_data
from my_packages.evaluation.code_evaluation import get_args_from_nodes
from my_packages.utils.file_utils import read_code_file, read_dataset_to_json

############################ Testing to Dynamically set the grammar file based on the nodes generated.
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

################ Testing regular grammar file parsing on dataset ###########################
with open("./midio_grammar.lark", "r") as f:
    grammar_text = f.read()
parser = Lark(grammar_text, parser="lalr", start="file", debug=True, strict=True)

main_dataset_folder = './MBPP_Midio_50/MBPP-Midio-50.json'
dataset = read_dataset_to_json(main_dataset_folder)
try:
    for i in range(0,50):
        # dataset[i]['code'] = dataset[i]['code'].replace("%%AVAILABLE_ARGS%%", available_args_union)
        i=i+1
        print(f"\nParsing code {i}...")
        code = read_code_file(i)
        # print(code)
        # Parse it
        tree = parser.parse(code)
        # Print the raw AST
        # print(tree.pretty())
        # print(list(parser.lex(code)))

    print("Parsing with Midio LARK grammar completed successfully on all samples in MBPP-Midio-50 dataset.")

except Exception as e:
    print(f"Error parsing code sample {i}: {e}")



