from lark import Lark
import sys
sys.path.append('../')  # Add the path to the my_packages module
from my_packages.utils.file_utils import read_code_file
reader = open("midio_grammar.lark", "r")
your_grammar = reader.read()
reader.close()
parser = Lark(your_grammar, parser="lalr", start="start")
code = read_code_file(1)
code = """
import("std", Std_k98ojb)
import("http", Http_q7o96c)

module() main {
    func(doc: "check whether the given list contains consecutive numbers or not.") check_consecutive {
        in(x: -425, y: 209, name: "list") property(List) list_faf6c2

        out(x: 866, y: 132, name: "output") property(Bool) output_a2b59d

        instance(x: -76, y: 64) filter_03f8ee root.Std_k98ojb.Iteration.Filter {
            handler: func() {
                in(x: -46, y: 13, name: "item") property(Number) item_7f9a85
                in(x: -41, y: 42, name: "index") property(Number) index_2b4b1a
                in(x: 11, y: -2, name: "list") property(List) list_4e8cf6

                out(x: 63, y: 40, name: "output_list") property(List) output_list_c9cfc8

                instance(x: 49, y: 17) expression_d81c1e root.Std_k98ojb.Math.Expression {
                    expression: "(index + 1) < list.length && item + 1 == list[index + 1]"
                }
                expression_d81c1e.result -> output_list_c9cfc8
            }
        }
        instance(x: 198, y: 89) not_3f3a62 root.Std_k98ojb.Logic.Not {}
        list_faf6c2 -> filter_03f8ee.items
        filter_03f8ee.output_list -> not_3f3a62.left
        not_3f3a62.result -> output_a2b59d
    }
    
    instance(x: -99, y: 141) check_consecutive_1f5a46 root.main.check_consecutive {}
}
    
"""
# Parse it
tree = parser.parse(code)

# Print the raw AST
print(tree.pretty())
print(list(parser.lex(code)))

