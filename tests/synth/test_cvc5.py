
"""Test for CVC5"""

from cvc5.pythonic import *

import cvc5
from cvc5 import Kind

# Get the string version of define-fun command.
# @param f the function to print
# @param params the function parameters
# @param body the function body
# @return a string version of define-fun


def define_fun_to_string(f, params, body):
    sort = f.getSort()
    if sort.isFunction():
        sort = f.getSort().getFunctionCodomainSort()
    result = "(define-fun " + str(f) + " ("
    for i in range(0, len(params)):
        if i > 0:
            result += " "
        result += "(" + str(params[i]) + " " + str(params[i].getSort()) + ")"
    result += ") " + str(sort) + " " + str(body) + ")"
    return result


# Print solutions for synthesis conjecture to the standard output stream.
# @param terms the terms for which the synthesis solutions were retrieved
# @param sols the synthesis solutions of the given terms


def print_synth_solutions(terms, sols):
    result = "(\n"
    for i in range(0, len(terms)):
        params = []
        body = sols[i]
        if sols[i].getKind() == Kind.LAMBDA:
            params += sols[i][0]
            body = sols[i][1]
        result += "  " + define_fun_to_string(terms[i], params, body) + "\n"
    result += ")"
    print(result)

class TestCVC5:
    def test_solver(self):
        # from https://cvc5.github.io/docs/latest/api/python/pythonic/quickstart.html
        # Let's introduce some variables
        # ! [docs-pythonic-quickstart-1 start]
        x, y = Reals('x y')
        a, b = Ints('a b')
        # ! [docs-pythonic-quickstart-1 end]

        # We will confirm that
        #  * 0 < x
        #  * 0 < y
        #  * x + y < 1
        #  * x <= y
        # are satisfiable
        # ! [docs-pythonic-quickstart-2 start]
        solve(0 < x, 0 < y, x + y < 1, x <= y)
        # ! [docs-pythonic-quickstart-2 end]

        # If we get the model (the satisfying assignment) explicitly, we can
        # evaluate terms under it.
        # ! [docs-pythonic-quickstart-3 start]
        s = Solver()
        s.add(0 < x, 0 < y, x + y < 1, x <= y)
        assert sat == s.check()
        m = s.model()
        # ! [docs-pythonic-quickstart-3 end]

        # ! [docs-pythonic-quickstart-4 start]
        print('x:', m[x])
        print('y:', m[y])
        print('x - y:', m[x - y])
        # ! [docs-pythonic-quickstart-4 end]

        # We can also get these values in other forms:
        # ! [docs-pythonic-quickstart-5 start]
        print('string x:', str(m[x]))
        print('decimal x:', m[x].as_decimal(4))
        print('fraction x:', m[x].as_fraction())
        print('float x:', float(m[x].as_fraction()))
        # ! [docs-pythonic-quickstart-5 end]

        # The above constraints are *UNSAT* for integer variables.
        # This reports "no solution"
        # ! [docs-pythonic-quickstart-6 start]
        solve(0 < a, 0 < b, a + b < 1, a <= b)

    def test_sygus(self):
        #from https://cvc5.github.io/docs/latest/examples/sygus-fun.html
        slv = cvc5.Solver()

        # required options
        slv.setOption("sygus", "true")
        slv.setOption("incremental", "false")

        # set the logic
        slv.setLogic("LIA")

        integer = slv.getIntegerSort()
        boolean = slv.getBooleanSort()

        # declare input variables for the functions-to-synthesize
        x = slv.mkVar(integer, "x")
        y = slv.mkVar(integer, "y")

        # declare the grammar non-terminals
        start = slv.mkVar(integer, "Start")
        start_bool = slv.mkVar(boolean, "StartBool")

        # define the rules
        zero = slv.mkInteger(0)
        one = slv.mkInteger(1)

        plus = slv.mkTerm(Kind.ADD, start, start)
        minus = slv.mkTerm(Kind.SUB, start, start)
        ite = slv.mkTerm(Kind.ITE, start_bool, start, start)

        And = slv.mkTerm(Kind.AND, start_bool, start_bool)
        Not = slv.mkTerm(Kind.NOT, start_bool)
        leq = slv.mkTerm(Kind.LEQ, start, start)

        # create the grammar object
        g = slv.mkGrammar([x, y], [start, start_bool])

        # bind each non-terminal to its rules
        g.addRules(start, [zero, one, x, y, plus, minus, ite])
        g.addRules(start_bool, [And, Not, leq])

        # declare the functions-to-synthesize. Optionally, provide the grammar
        # constraints
        max = slv.synthFun("max", [x, y], integer, g)
        min = slv.synthFun("min", [x, y], integer)

        # declare universal variables.
        varX = slv.declareSygusVar("x", integer)
        varY = slv.declareSygusVar("y", integer)

        max_x_y = slv.mkTerm(Kind.APPLY_UF, max, varX, varY)
        min_x_y = slv.mkTerm(Kind.APPLY_UF, min, varX, varY)

        # add semantic constraints
        # (constraint (>= (max x y) x))
        slv.addSygusConstraint(slv.mkTerm(Kind.GEQ, max_x_y, varX))

        # (constraint (>= (max x y) y))
        slv.addSygusConstraint(slv.mkTerm(Kind.GEQ, max_x_y, varY))

        # (constraint (or (= x (max x y))
        #                 (= y (max x y))))
        slv.addSygusConstraint(slv.mkTerm(
            Kind.OR,
            slv.mkTerm(Kind.EQUAL, max_x_y, varX),
            slv.mkTerm(Kind.EQUAL, max_x_y, varY)))

        # (constraint (= (+ (max x y) (min x y))
        #                (+ x y)))
        slv.addSygusConstraint(slv.mkTerm(
            Kind.EQUAL,
            slv.mkTerm(Kind.ADD, max_x_y, min_x_y),
            slv.mkTerm(Kind.ADD, varX, varY)))

        # print solutions if available
        if (slv.checkSynth().hasSolution()):
            # Output should be equivalent to:
            # (define-fun max ((x Int) (y Int)) Int (ite (<= x y) y x))
            # (define-fun min ((x Int) (y Int)) Int (ite (<= x y) x y))
            terms = [max, min]
            print_synth_solutions(terms, slv.getSynthSolutions(terms))
