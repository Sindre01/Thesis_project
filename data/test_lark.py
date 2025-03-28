from lark import Lark
import sys
sys.path.append('../')  # Add the path to the my_packages module
from my_packages.utils.file_utils import read_code_file
reader = open("midio_grammar.lark", "r")
your_grammar = reader.read()
reader.close()
parser = Lark(your_grammar, parser="lalr", start="start")
code = read_code_file(1)
# Parse it
tree = parser.parse(code)

# Print the raw AST
print(tree.pretty())
print(list(parser.lex(code)))

