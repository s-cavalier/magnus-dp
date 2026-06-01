#include "gausslegendre.hpp"
#include <stdexcept>

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