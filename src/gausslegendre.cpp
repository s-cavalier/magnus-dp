#include "gausslegendre.hpp"
#include <stdexcept>
#include <utility>

Magnus::GLTable::GLTable(size_t max_order) : m_max_order(max_order) {}

Magnus::GLTable::DataView::DataView(const double* weights, const double* nodes, size_t order) : m_w(weights), m_x(nodes), n(order) {}

size_t Magnus::GLTable::DataView::order() const {
    return n;
}

std::pair<double, double> Magnus::GLTable::DataView::operator[](size_t i) const {
    return { m_w[i], m_x[i] };
}

std::pair<double, double> Magnus::GLTable::DataView::at(size_t i) const {
    if ( i >= n ) throw std::out_of_range("Invalid access of GLTable::DataView::at(size_t) const");
    return { m_w[i], m_x[i] };
}

const double* Magnus::GLTable::DataView::weights() const {
    return m_w;
}

const double* Magnus::GLTable::DataView::nodes() const {
    return m_x;
}

size_t Magnus::GLTable::max_order() const { return m_max_order; }

Magnus::GLTable::DataView Magnus::StaticTable::get_order(size_t n) const {
    if ( n == 0 || n > max_order() ) {
        throw std::out_of_range("Gauss-Legendre order is outside the available table");
    }

    size_t offset = offsets[n];

    return DataView{
        weights + offset,
        nodes + offset,
        n
    };
}

Magnus::OwningTable::OwningTable(
    size_t max_order,
    std::vector<size_t> offsets_,
    std::vector<double> weights_,
    std::vector<double> nodes_
) :
    GLTable(max_order),
    offsets(std::move(offsets_)),
    weights(std::move(weights_)),
    nodes(std::move(nodes_))
{
    if ( max_order == 0 ) throw std::invalid_argument("OwningTable requires max_order >= 1");
    if ( offsets.size() != max_order + 1 ) throw std::invalid_argument("OwningTable offsets size mismatch");
    if ( weights.size() != nodes.size() ) throw std::invalid_argument("OwningTable node/weight size mismatch");

    size_t expected_offset = 0;
    if ( offsets[0] != 0 ) throw std::invalid_argument("OwningTable offsets must start at zero");

    for ( size_t order = 1; order <= max_order; ++order ) {
        if ( offsets[order] != expected_offset ) {
            throw std::invalid_argument("OwningTable offsets are not contiguous by order");
        }

        expected_offset += order;
    }

    if ( weights.size() != expected_offset ) {
        throw std::invalid_argument("OwningTable flattened node/weight size mismatch");
    }
}

Magnus::GLTable::DataView Magnus::OwningTable::get_order(size_t n) const {
    if ( n == 0 || n > max_order() ) {
        throw std::out_of_range("Gauss-Legendre order is outside the available table");
    }

    size_t offset = offsets[n];

    return DataView{
        weights.data() + offset,
        nodes.data() + offset,
        n
    };
}
