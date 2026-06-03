from __future__ import annotations

import yaml
from dataclasses import dataclass
import re

# from spec.yaml:
# Axis spec:
# Any code must work within the context of only importing `manual_impls.hpp`.
# A fun benefit of this is code specified via manual_impls.hpp can be used in requires/when expressions below.
# Any code involving {num_t} will use a std::variant wrapper to properly dispatch, so num_t can be abstracted away.
# Furthermore, all Parameter fields can be accessed via {field}. 
#
# axis: !Axis
#   enum_t: str, Emitted enum type
#   values: list of {id, cpp, optional<requires>}, specifying different possible dispatches
#     id: str, specifies the name of the enum
#     cpp: str, specifies the C++ dispatch routine
#     optional<requires>: str | None, if specified, is an expression declaring a mandatory condition. An exception is thrown if this fails.
#   select: list of { use, optional<when> }, specifies the deduction order if user does not specify at call-site. Goes top-down, follows a if-else setup (for example, the last `use` does not need a `when`, but every other `use` needs a `when` unless a `requires` is defined). Must be the last item defined in an axis.
#     use: str, specifies the id to dispatch to if `when` is valid. Must be a previously specified id, will otherwise throw.
#     when: optional<str>, specifies an expression declaring if it should be selected. If a requires expression exists for id, it will be included as part of the deduction rules, so respecifying the same expr as in `requires` is okay, but unnecessary (will result in ( (requires_expr) && (when_expr) )). Thus if a requires exist, a `when` can be ignored. Similarly, if the last item, a `when` can be ignored. If the last item does not have a `when`, there will be a C++ exception thrown if nothing is valid.



@dataclass
class Axis(yaml.YAMLObject):

    @dataclass
    class Expr:
        data: str

        NUM_T_PATTERN = re.compile(r"\bnum_t\b")

        def uses_num_t(self) -> bool:
            return self.NUM_T_PATTERN.search(self.data) is not None
        
        def emit(self) -> str:
            return data

    @dataclass
    class Value:
        id: str
        prefix: str
        cpp: Axis.Expr
        requires: Axis.Expr | None = None

        def emit(self) -> str:
            if self.requires is None: return f"case {self.prefix}::{self.id}: return {self.prefix}::{self.id};"

            return f"""case {self.prefix}::{self.id}:
                if ( !({self.requires}) ) throw std::invalid_argument("Cannot dispatch to {self.id} under given options");
                return {self.prefix}::{self.id};
            """

    @dataclass
    class Select:
        use: str
        prefix: str
        when: Axis.Expr | None = None
        requires: Axis.Expr | None = None

        def emit(self) -> str:
            if self.requires is None and self.when is None: return f"return {self.prefix}::{self.use};"
            if self.requires is not None and self.when is not None: return f"if (({self.requires.emit()}) && ({self.when.emit()})) return {self.prefix}::{self.use};"
            if self.when is not None: return f"if ({self.when.emit()}) return {self.prefix}::{self.use};"
            return f"if ({self.requires.emit()}) return {self.prefix}::{self.use};"

    yaml_tag = "!Axis"
    yaml_loader = yaml.Loader

    enum_t: str
    values: dict[str, Value]
    select: list[Select]

    @classmethod
    def from_yaml(cls, loader, node):
        d = loader.construct_mapping(node, deep=True)
        enum_t = d['enum_t']
        values = { v['id'] : Axis.Value(id=v['id'], prefix=enum_t, cpp=Axis.Expr(v['id']), requires= None if 'requires' not in v else Axis.Expr(v['requires']) ) for v in d['values']}
        select = [None] * len(values)
        for i, s in enumerate(d['select']):
            use = s['use']
            if use not in values: raise TypeError(f"{use} was specified in a selector but not as an enum value")
            select[i] = Axis.Select( use=use, prefix=enum_t, when = None if 'when' not in s else Axis.Expr(s['when']) , requires = values[use].requires )

        return cls( enum_t, values, select )

    def card(self) -> int:
        return len(self.values)

    def emit_enum(self) -> str:
        return f"enum class {self.enum_t} {{ {', '.join(value.id for value in self.values.values())}, Auto }};"

    def emit_resolver(self) -> str:
        return f"""{self.enum_t} {self.enum_t}_resolver({self.enum_t} req, Parameters& param) {{
            switch (req) {{
                { '\n'.join( value.emit() for value in self.values.values() ) }
                default:
                { '\n'.join( select.emit() for select in self.select ) }
            }}
        }}"""


with open("spec.yaml") as f: data = yaml.load(f, Loader=yaml.Loader)

axis : Axis = data['axes']['matrix_policy']

print( axis )
print()
print( axis.emit_enum() )
print()
print( axis.emit_resolver() )
