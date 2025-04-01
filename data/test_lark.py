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
import("std", Std_k98ojb)
import("http", Http_q7o96c)

module() main { 
    func(doc: "Finds the smallest missing number from a sorted list of natural numbers.") find_First_Missing {
        in(x: -113, y: 62, name: "list") property(List) list_24e9a6
        in(x: 483, y: -328, name: "execute") trigger() execute_fa8807

        out(x: 1453, y: 319, name: "output") property(Number) output_25655e
        out(x: 1491, y: -61, name: "continue") trigger() continue_aedf0f

        instance(x: -233, y: -173) for_0e7c92 root.Std_k98ojb.Std.For {
            reset: "execute"
        }
        instance(x: -3, y: 87) if_4f8c39 root.Std_k98ojb.Std.If {
            input: "item != index + 1"
        }
        instance(x: 148, y: -84) ifexpression_1e2a0a root.Std_k98ojb.Std.IfExpression {
            expression: "index == 0 && item != 1"
        }
        instance(x: 147, y: 6) ifexpression_7f2e76 root.Std_k98ojb.Std.IfExpression {
            expression: "index > 0 && (item - last_item) > 1"
        }
        getter(x: 103, y: 127, name: "getter_0b7c6d") getter_0b7c6d = output
        setter(x: 107, y: -52, name: "setter_4b5f0d") setter_4b5f0d = output
        setter(x: 93, y: -269, name: "setter_1d4a6c") setter_1d4a6c = output
        getter(x: 102, y: 9, name: "getter_0f7a3f") getter_0f7a3f = last_item
        setter(x: 97, y: -142, name: "setter_4a7e8c") setter_4a7e8c = last_item
        list_24e9a6 -> for_0e7c92.items
        for_0e7c92.item -> if_4f8c39.input
        for_0e7c92.index -> ifexpression_1e2a0a.gen_1
        for_0e7c92.index -> ifexpression_7f2e76.gen_1
        ifexpression_1e2a0a.gen_0 -> ifexpression_1e2a0a.gen_2
        ifexpression_7f2e76.gen_0 -> ifexpression_7f2e76.gen_2
        getter_0f7a3f.value -> ifexpression_1e2a0a.gen_0
        getter_0f7a3f.value -> ifexpression_7f2e76.gen_0
        ifexpression_1e2a0a.result -> if_4f8c39.input
        ifexpression_7f2e76.result -> if_4f8c39.input
        for_0e7c92.on_item -> if_4f8c39.execute
        execute_fa8807 -> for_0e7c92.trigger_
        instance_ -> if_4f8c39.execute
    }
}
    
"""
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


