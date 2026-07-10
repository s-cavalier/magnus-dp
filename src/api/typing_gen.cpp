#include "composer/backends.hpp"

#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace {

template <class BackendList>
void emit_literal_alias(std::ostream& out, std::string_view name, bool include_auto = false) {
    out << name << " = Literal[";

    bool first = true;
    for (std::string_view backend_name : BackendList::dispatch_names) {
        if (!first) out << ", ";
        out << std::quoted(backend_name);
        first = false;
    }

    if (include_auto) {
        if (!first) out << ", ";
        out << std::quoted("Auto");
    }

    out << "]\n";
}

template <class Names>
void emit_literal_alias_from_names(std::ostream& out, std::string_view name, const Names& names) {
    out << name << " = Literal[";

    bool first = true;
    for (std::string_view backend_name : names) {
        if (!first) out << ", ";
        out << std::quoted(backend_name);
        first = false;
    }

    out << "]\n";
}

void write_typing_file(std::ostream& out) {
    out << "from typing import Literal\n\n";
    emit_literal_alias<Magnus::NumBackends>(out, "NumericBackendName");
    emit_literal_alias<Magnus::MatrixBackends>(out, "MatrixBackendName", true);
    emit_literal_alias<Magnus::IntegratorBackends>(out, "IntegratorName", true);
    emit_literal_alias_from_names(out, "KernelOpName", Magnus::Dispatch::kernel_op_names);
}

}

int main(int argc, char** argv) {
    if (argc != 2) {
        throw std::invalid_argument("usage: typing_gen <output-file>");
    }

    std::ofstream out(argv[1], std::ios::binary);
    if (!out) {
        throw std::runtime_error("could not open output file");
    }

    write_typing_file(out);
    return 0;
}
