#ifndef __INTEGRATE_HPP__
#define __INTEGRATE_HPP__
#include <span>
#include <utility>
#include <memory>
#include <atomic>

/*
An abstract handle onto some GaussLegendre table.
We use a global PIMPL approach so that way the details of "where" the data is stored is abstracted away.
It could be a statically linked GL table, mmapped table, heap table, or even just a file.

Assumes doubles. Maybe extend later for other purposes.
*/
class GLTable {
    inline static std::atomic<std::shared_ptr<const GLTable>> global_table{nullptr};

protected:
    size_t m_max_order;
    GLTable(size_t max_order);

public:
    class DataView {
        const double* m_w;
        const double* m_x;
        size_t n;

    public:
        DataView(const double* weights, const double* nodes, size_t order);

        size_t order() const;
        std::pair<double, double> operator[](size_t i) const;
        std::pair<double, double> at(size_t i) const;

        const double* weights() const;
        const double* nodes() const;
    };

    virtual ~GLTable() = default;
    virtual DataView get_order(size_t n) const = 0;

    size_t max_order() const;

    static std::shared_ptr<const GLTable> get() {
        auto ptr = global_table.load(std::memory_order_acquire);

        if (!ptr) throw std::runtime_error("Global Gauss-Legendre table is null");

        return ptr;
    }

    static void update(std::shared_ptr<const GLTable> new_table) {
        if (!new_table) throw std::invalid_argument("Tried to initialize table with nullptr");
        global_table.store( std::move(new_table), std::memory_order_release );
    }

};

class StaticTable : public GLTable {
    struct PyGLTableHeader {
        char magic[4];
        uint32_t max_n;

        size_t offsets_offset;
        size_t nodes_offset;
        size_t weights_offset;
    };

    size_t* offsets;
    double* weights;
    double* nodes;

public:
    StaticTable(std::byte* table) : GLTable(0) {
        PyGLTableHeader* header = (PyGLTableHeader*)table;

        if ( header->magic[0] != 'L' || header->magic[1] != 'G' || header->magic[2] != '0' || header->magic[3] != '1' )
            throw std::runtime_error("Invalid magic number; StaticTable(std::byte*)");

        this->m_max_order = header->max_n;

        offsets = (size_t*)(table + header->offsets_offset);
        weights = (double*)(table + header->weights_offset);
        nodes = (double*)(table + header->nodes_offset);
    }

    DataView get_order(size_t n) const override {
        size_t offset = offsets[n];

        return DataView{
            weights + offset,
            nodes + offset,
            n
        };
    }

};


#endif