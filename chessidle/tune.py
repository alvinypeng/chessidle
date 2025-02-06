from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
import math
from typing import TYPE_CHECKING

import chessidle.history as history
import chessidle.move_pick as move_pick
import chessidle.search as search
from chessidle.options import Options, OptionRange


FLOAT_SCALE = 1000


modules = search, move_pick, history

def generate_parameter_name(module, node) -> str:
    module_name = module.__name__.replace('chessidle.', '')
    return f'TUNE[{module_name}][line:{node.lineno}][col:{node.col_offset}][{node.value}]'
    

_options_info = []

for module in modules:
    try:
        tree = ast.parse(inspect.getsource(module))
    except OSError:
        break

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            # Don't tune any zeros or ones.
            if isinstance(node.value, int) and node.value in (0, 1):
                continue
            
            name = generate_parameter_name(module, node)
            default = int(node.value * FLOAT_SCALE) if isinstance(node.value, float) else node.value
            minimum = 0
            maximum = 2 * default
            option_range = OptionRange(int, default, minimum, maximum)
            
            _options_info.append(
                (module.__name__, node.lineno, node.col_offset, name, option_range)
            )


TUNE_OPTION_RANGES = {
    name: option_range
    for _, _, _, name, option_range in sorted(_options_info)
}

TUNE_OPTION_DEFAULTS = {key: value.default for key, value in TUNE_OPTION_RANGES.items()}

CONFIG_STRING = json.dumps(
    {
        name: {
            'value': option_range.default,
            'min_value': option_range.minimum,
            'max_value': option_range.maximum,
            'step': math.ceil(option_range.default / 10),
        }
        for name, option_range in TUNE_OPTION_RANGES.items()
    },
    indent=4,
)


def modified_code(options: Options) -> str:
    code = ''
    
    for module in modules:
        tree = ast.parse(inspect.getsource(module))
        
        class ConstantModifier(ast.NodeTransformer):
                
            def visit_Constant(self, node):
                if type(node.value) not in (int, float):
                    return node

                if isinstance(node.value, int) and node.value in (0, 1):
                    return node
                    
                name = generate_parameter_name(module, node)
                value = options[name]

                if isinstance(node.value, float):
                    value /= FLOAT_SCALE

                return ast.Constant(value=value)

        modifier = ConstantModifier()
        new_tree = modifier.visit(copy.deepcopy(tree))
        code += ast.unparse(new_tree) + '\n'

    # Remove multiple __future__ imports.
    future_import_statement = 'from __future__ import annotations\n'
    code = future_import_statement + code.replace(future_import_statement, '')

    return code
