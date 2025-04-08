import time
from lark import Lark
import sys
sys.path.append('../')  # Add the path to the my_packages module
from my_packages.utils.file_utils import read_code_file
reader = open("midio_grammar.lark", "r")
your_grammar = reader.read()
reader.close()
parser = Lark(your_grammar, parser="lalr", start="start", debug=True, strict=True)

code = """
import("std", Std_k98ojb)\nimport("http", Http_q7o96c)\n\nmodule() main { \n func(doc: "gets the sum of the digits of a non-negative integer.") sum_digits{\n in(x: -450, y: -421, name: "execute") trigger() execute_cdac2a\n in(x: -436, y: -213, name: "n") property(Number) n_6b655b\n out(x: 1146, y: -647, name: "continue") trigger() continue_d9efd7\n out(x: 1169, y: -269, name: "output") property(Number) output_732a8a\n\n instance(x: 17, y: -417, name: "sum_digits") Std_k98ojb Std.IfExpression {\n expression: "Std.Count([Std.Characters(n_6b655b)]) > 0"\n }\n n_6b655b -> sum_digits.input\n sum_digits.then -> Std_k98ojb.Std.For {\n in x: -240, y: 69, name: "string" property(String) string_4f6d2a\n in x: 177, y: 58, name: "count" property(Number) count_1f468d\n in x: 723, y: 62, name: "characters" property(List) characters_7942d4\n in x: 46, y: 66, name: "filter" property(List) filter_c1bb22\n in x: 62, y: 62, name: "length" property(Number) length_8815e6\n in x: 24, y: -48, name: "current" property(Number) current_86a8fb\n in x: 74, y: 2, name: "isnumeric" property(Bool) isnumeric_9cb360\n in x: 298, y: 1, name: "expression" property(String) expression_634b7e\n in x: 818, y: -74, name: "add" property(Number) add_8974a2\n in x: 2, y: 1, name: "mul" property(Number) mul_0f3be6\n in x: 1, y: -74, name: "sub" property(Number) sub_2c4e5d\n in x: 0, y: 1, name: "result" property(Number) result_8d8f8d\n in x: 0, y: 0, name: "output" property(Number) output_8ba733\n in x: 14, y: -52, name: "number_ctr" property(Number) number_ctr_407e1b\n in x: 723, y: 62, name: "characters" -> filter_c1bb22.items\n in x: 62, y: 62, name: "length" -> filter_c1bb22.length\n in x: 24, y: -48, name: "current" -> filter_c1bb22.current\n in x: 74, y: 2, name: "isnumeric" -> filter_c1bb22.isnumeric\n in x: 298, y: 1, name: "expression" -> filter_c1bb22.expression\n in x: 818, y: -74, name: "add" -> filter_c1bb22.add\n in x: 2, y: 1, name: "mul" -> filter_c1bb22.mul\n in x: 1, y: -74, name: "sub" -> filter_c1bb22.sub\n in x: 0, y: 1, name: "result" -> filter_c1bb22.output\n in x: 0, y: 0, name: "output" -> filter_c1bb22.output_list\n in x: 14, y: -52, name: "number_ctr" -> filter_c1bb22.on_item\n in x: 723, y: 62, name: "characters" -> filter_c1bb22.on_item\n in x: 62, y: 62, name: "length" -> filter_c1bb22.on_item\n in x: 24, y: -48, name: "current" -> filter_c1bb22.on_item\n in x: 74, y: 2, name: "isnumeric" -> filter_c1bb22.on_item\n in x: 298, y: 1, name: "expression" -> filter_c1bb22.on_item\n in x: 818, y: -74, name: "add" -> filter_c1bb22.on_item\n in x: 2, y: 1, name: "mul" -> filter_c1bb22.on_item\n in x: 1, y: -74, name: "sub" -> filter_c1bb22.on_item\n in x: 0, y: 1, name: "result" -> filter_c1bb22.on_item\n in x: 0, y: 0, name: "output" -> filter_c1bb22.on_item\n in x: 14, y: -52, name: "number_ctr" -> filter_c1bb22.on_item\n in x: 723, y: 62, name: "characters" -> filter_c1bb22.on_item\n in x: 62, y: 62, name: "length" -> filter_c1bb22.on_item\n in x: 24, y: -48, name: "current" -> filter_c1bb22.on_item\n in x: 74, y: 2, name: "isnumeric" -> filter_c1bb22.on_item\n in x: 298, y: 1, name: "expression" -> filter_c1bb22.on_item\n in x: 818, y: -74, name: "add" -> filter_c1bb22.on_item\n in x: 2, y: 1, name: "mul" -> filter_c1bb22.on_item\n in x: 1, y: -74, name: "sub" -> filter_c1bb22.on_item\n in x: 0, y: 1, name: "result" -> filter_c1bb22.on_item\n in x: 0, y: 0, name: "output" -> filter_c1bb22.on_item\n in x: 14, y: -52, name: "number_ctr" -> filter_c1bb22.on_item\n in x: 723, y: 62, name: "characters" -> filter_c1bb22.on_item\n in x: 62, y: 62, name: "length" -> filter_c1bb22.on_item\n in x: 24, y: -48, name: "current" -> filter_c1bb22.on_item\n in x: 74, y: 2, name: "isnumeric" -> filter_c1bb22.on_item\n in x: 298, y: 1, name: "expression" -> filter_c1bb22.on_item\n in x: 818, y: -74, name: "add" -> filter_c1bb22.on_item\n in x: 2, y: 1, name: "mul" -> filter_c1bb22.on_item\n in x: 1, y: -74, name: "sub" -> filter_c1bb22.on_item\n in x: 0, y: 1, name: "result" -> filter_c1bb22.on_item\n in x: 0, y: 0, name: "output" -> filter_c1bb22.on_item\n in x: 14, y: -52, name: "number_ctr" -> filter_c1bb22.on_item\n in x: 723, y: 62, name: "characters" -> filter_c1bb22.on_item\n in x: 62, y: 62, name: "length" -> filter_c1bb22.on_item\n in x: 24, y: -48, name: "current" -> filter_c1bb22.on_item\n in x: 74, y: 2, name: "isnumeric" -> filter_c1bb22.on_item\n in x: 298, y: 1, name: "expression" -> filter_c1bb22.on_item\n in x: 818, y: -74, name: "add" -> filter_c1bb22.on_item\n in x: 2, y: 1, name: "mul" -> filter_c1bb22.on_item\n in x: 1, y: -74, name: "sub" -> filter_c1bb22.on_item\n in x: 0, y: 1, name: "result" -> filter_c1bb22.on_item\n in x: 0, y: 0, name:
    
"""
print(code)
# tree = parser.parse(code)

# # Print the raw AST
# print(tree.pretty())
print(list(parser.lex(code)))
time.sleep(2)
# for i in range(0,50):
#     i=i+1
#     print(f"\n\nParsing code {i}...")
#     code = read_code_file(i)
#     print(code)
#     # Parse it
#     tree = parser.parse(code)

#     # Print the raw AST
#     # print(tree.pretty())
#     print(list(parser.lex(code)))


