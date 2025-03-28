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

mask = """
    import("std", Std_k98ojb)\nimport("http", Http_q7o96c)\n\nmodule() main { \n    func(doc: "check whether the entered number is greater than the elements of the given array.") check_greater {\n        in(x: -285, y: 82, name: "list") property(List) list_1a1743\n        in(x: -285, y: -8, name: "num") property(Number) num_1a1743\n\n        out(x: 197, y: -8, name: "output") property(Bool) output_4ff980\n\n        instance(x: -203, y: -7,) filter_0e15bd root.Std_k98ojb.Query.Filter {\n            where: "it < num"\n        }\n        instance(x: -2, y: -7,) not_3c0cb root.Std_k98ojb.Logic.Not {}\n        list_1a1743 -> filter_0e15bd.list\n        num_1a1743 -> filter_0e15bd.gen_0\n        filter_0e15bd.result -> not_3c0cb.input\n        not_3c0cb.output -> output_4ff980\n    }\n\n    \n\n    instance(x: 80, y: 103) check_greater_db84c5 root.main.check_greater {}\n\n}\n\n    \n    import("std", Std_k98ojb)\nimport("http", Http_q7o96c)\n\nmodule() main { \n    func(doc: "check whether the entered number is greater than the elements of the given array.") check_greater {\n        in(x: -285, y: 82, name: "list") property(List) list_1a1743\n        in(x: -285, y: -8, name: "num") property(Number) num_1a1743\n\n        out(x: 197, y: -8, name: "output") property(Bool) output_4ff980\n\n        instance(x: -203, y: -7,) filter_0e15bd root.Std_k98ojb.Query.Filter {\n            where: "it < num"\n        }\n        instance(x: -2, y: -7,) not_3c0cb root.Std_k98ojb.Logic.Not {}\n        list_1a1743 -> filter_0e15bd.list\n        num_1a1743 -> filter_0e15bd.gen_0\n        filter_0e15bd.result -> not_3c0cb.input\n        not_3c0cb.output -> output_4ff980\n    }\n\n    \n\n    instance(x: 80, y: 103) check_greater_db84c5 root.main.check_greater {}\n\n}\n\n    \n    import("std", Std_k98ojb)\nimport("http", Http_q7o96c)\n\nmodule() main { \n    func(doc: "check whether the entered number is greater than the elements of the given array.") check_greater {\n        in(x: -285, y: 82, name: "list") property(List) list_1a1743\n        in(x: -285, y: -8, name: "num") property(Number) num_1a1743\n\n        out(x: 197, y: -8, name: "output") property(Bool) output_4ff980\n\n        instance(x: -203, y: -7,) filter_0e15bd root.Std_k98ojb.Query.Filter {\n            where: "it < num"\n        }\n        instance(x: -2, y: -7,) not_3c0cb root.Std_k98ojb.Logic.Not {}\n        list_1a1743 -> filter_0e15bd.list\n        num_1a1743 -> filter_0e15bd.gen_0\n        filter_0e15bd.result -> not_3c0cb.input\n        not_3c0cb.output -> output_4ff980\n    }\n\n    \n\n    instance(x: 80, y: 103) check_greater_db84c5 root.main.check_greater {}\n\n}\n\n    \n    import("std", Std_k98ojb)\nimport("http", Http_q7o96c)\n\nmodule() main { \n    func(doc: "check whether the entered number is greater than the elements of the given array.") check_greater {\n        in(x: -285, y: 82, name: "list") property(List) list_1a1743\n        in(x: -285, y: -8, name: "num") property(Number) num_1a1743\n\n        out(x: 197, y: -8, name: "output") property(Bool) output_4ff980\n\n        instance(x: -203, y: -7,) filter_0e15bd root.Std_k98ojb.Query.Filter {\n            where: "it < num"\n        }\n        instance(x: -2, y: -7,) not_3c0cb root.Std_k98ojb.Logic.Not {}\n        list_1a1743 -> filter_0e15bd.list\n        num_1a1743 -> filter_0e15bd.gen_0\n        filter_0e15bd.result -> not_3c0cb.input\n        not_3c0cb.output -> output_4ff980\n    }\n\n    \n\n    instance(x: 80, y: 103) check_greater_db84c5 root.main.check_greater {}\n\n}\n\n    \n    import("std", Std_k98ojb)\nimport("http", Http_q7o96c)\n\nmodule() main { \n    func(doc: "check whether the entered number is greater than the elements of the given array.") check_greater {\n        in(x: -285, y: 82, name: "list") property(List) list_1a1743\n        in(x: -285, y: -8, name: "num") property(Number) num_1a1743\n\n        out(x: 197, y: -8, name: "output") property(Bool) output_4ff980\n\n        instance(x: -203, y: -7,) filter_0e15bd root.Std_k98ojb.Query.Filter {\n            where: "it < num"\n        }\n        instance(x: -2, y: -7,) not_3c0cb root.Std_k98ojb.Logic.Not {}\n        list_1a1743 -> filter_0e15bd.list\n        num_1a1743 -> filter_0e15bd.gen_0\n        filter_0e15bd.result -> not_3c0cb.input\n        not_3c0cb.output -> output_4ff980\n    }\n\n    \n\n    instance(x: 80, y: 103) check_greater_db84c5 root.main.check_greater {}\n\n}\n\n    \n    import("std", Std_k98ojb)\nimport("http", Http_q7o96c)\n\nmodule() main { \n    func(doc: "check whether the entered number is greater than the elements of the given array.") check_greater {\n        in(x: -285, y: 82, name: "list") property(List) list_1a1743\n        in(x: -285, y: -8, name: "num") property(Number) num_1a1743\n\n        out(x: 197, y: -8, name: "output") property(Bool) output_4ff980\n\n        instance(x: -203, y: -7,) filter_0e15bd root.Std_k98ojb.Query.Filter {\n            where: "it < num"\n        }\n        instance(x: -2, y: -7,) not_3c0cb root.Std_k98ojb.Logic.Not {}\n        list_1a1743 -> filter_0e15bd.list\n        num_1a1743 -> filter_0e15bd.gen_0\n        filter_0e15bd.result -> not_3c0cb.input\n        not_3c0cb.output -> output_4ff980\n    }\n\n    \n\n    instance(x: 80, y: 103) check_greater_db84c5 root.main.check_greater {}\n\n}\n\n    \n    import("std", Std_k98ojb)\nimport("http", Http_q7o96c)\n\nmodule() main { \n    func(doc: "check whether the entered number is greater than the elements of the given array.") check_greater {\n        in(x: -285, y: 82, name: "list") property(List) list_1a1743\n        in(x: -285, y: -8, name: "num") property(Number) num_1a1743\n\n        out(x: 197, y: -8, name: "output") property(Bool) output_4ff980\n\n        instance(x: -203, y: -7,) filter_0e15bd root.Std_k98ojb.Query.Filter {\n            where: "it < num"\n        }\n        instance(x: -2, y: -7,) not_3c0cb root.Std_k98ojb.Logic.Not {}\n        list_1a1743 -> filter_0e15bd.list\n        num_1a1743 -> filter_0e15bd.gen_0\n        filter_0e15bd.result -> not_3c0cb.input\n        not_3c0cb.output -> output_4ff980\n    }\n\n    \n\n    instance
"""
print(mask)
orginal = """
import("std", Std_k98ojb)
import("http", Http_q7o96c)

        module() main {
            func(doc: "Returns true if x > y") is_greater {
           in(x: -285, y: 82, name: "num") property(Number) num_1a1743
        in(x: -285, y: -8, name: "execute") trigger() execute_1a1743

        out(x: 197, y: -8, name: "continue") trigger() continue_4ff980
        out(x: 197, y: -8, name: "output") property(Bool) output_4ff980

        instance(x: -203, y: -7,) for_0e15bd root.Std_k98ojb.Std.For {}
        instance(x: -2, y: -7,) if_0e15bd root.Std_k98ojb.Std.If {}
        instance(x: -2, y: -7,) mod_0e15bd root.Std_k98ojb.Math.Modulo {}
        instance(x: -2, y: -7,) expression_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n % i == 0"
        }
        instance(x: -2, y: -7,) expression_2_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "i == 1"
        }
        instance(x: -2, y: -7,) expression_3_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 2"
        }
        instance(x: -2, y: -7,) expression_4_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 1"
        }
        instance(x: -2, y: -7,) expression_5_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 0"
        }
        instance(x: -2, y: -7,) expression_6_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 3"
        }
        instance(x: -2, y: -7,) expression_7_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 4"
        }
        instance(x: -2, y: -7,) expression_8_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 5"
        }
        instance(x: -2, y: -7,) expression_9_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 6"
        }
        instance(x: -2, y: -7,) expression_10_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 7"
        }
        instance(x: -2, y: -7,) expression_11_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 8"
        }
        instance(x: -2, y: -7,) expression_12_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 9"
        }
        instance(x: -2, y: -7,) expression_13_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 10"
        }
        instance(x: -2, y: -7,) expression_14_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 11"
        }
        instance(x: -2, y: -7,) expression_15_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 12"
        }
        instance(x: -2, y: -7,) expression_16_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 13"
        }
        instance(x: -2, y: -7,) expression_17_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 14"
        }
        instance(x: -2, y: -7,) expression_18_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 15"
        }
        instance(x: -2, y: -7,) expression_19_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 16"
        }
        instance(x: -2, y: -7,) expression_20_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 17"
        }
        instance(x: -2, y: -7,) expression_21_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 18"
        }
        instance(x: -2, y: -7,) expression_22_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 19"
        }
        instance(x: -2, y: -7,) expression_23_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 20"
        }
        instance(x: -2, y: -7,) expression_24_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 21"
        }
        instance(x: -2, y: -7,) expression_25_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 22"
        }
        instance(x: -2, y: -7,) expression_26_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 23"
        }
        instance(x: -2, y: -7,) expression_27_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "n == 24"
        }
        instance(x: -2, y: -7,) expression_28_0e15bd root.Std_k98ojb.Math.Expression {
            expression: "

"""
print(orginal)

