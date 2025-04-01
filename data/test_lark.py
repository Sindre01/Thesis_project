import time
from lark import Lark
import sys
sys.path.append('../')  # Add the path to the my_packages module
from my_packages.utils.file_utils import read_code_file
reader = open("midio_grammar.lark", "r")
your_grammar = reader.read()
reader.close()
parser = Lark(your_grammar, parser="lalr", start="start")

code = """
import("std", Std_k98ojb)
import("http", Http_q7o96c)

module() main {
    func(doc: "Calculates the product of the unique numbers in a given list.") unique_product {
        in(x: -757, y: -167, name: "execute") trigger() execute_19300c
        in(x: -241, y: 24, name: "list") property(List) list_5fda54
        out(x: 887, y: -144, name: "continue") trigger() continue_45190b
        out(x: 683, y: 73, name: "output") property(Number) output_cffcc2
    }
}
    
"""
for i in range(0,50):
    i=i+1
    print(f"\n\nParsing code {i}...")
    code = read_code_file(i)
    print(code)
    # Parse it
    tree = parser.parse(code)

    # Print the raw AST
    # print(tree.pretty())
    print(list(parser.lex(code)))
    time.sleep(2)

