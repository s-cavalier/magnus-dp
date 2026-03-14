#include <iostream>
#include <ranges>
#include <span>
#include <string_view>
#include <memory>
#include <fcntl.h>
#include "mag_dispatch.h"

void early_exit_test( bool condition, const char* msg ) {
    if (condition) return;
    std::cerr << msg;
    exit(1);
}

bool parse_ulong(const char* str, unsigned long& result) {
    errno = 0;

    char* end;
    unsigned long value = std::strtoul(str, &end, 10);

    if (str == end) return false; 
    if (*end != '\0') return false; 
    if (errno == ERANGE) return false; 

    result = value;
    return true;
}

bool parse_double(const char* str, double& result) {
    errno = 0;

    char* end;
    double value = std::strtod(str, &end);

    if (str == end) return false; 
    if (*end != '\0') return false; 
    if (errno == ERANGE) return false; 

    result = value;
    return true;
}

int main(int argc, char** argv) {
    // expected: ./main <degree> <path to input_buffer>
    early_exit_test( argc == 5, "USAGE: main <degree> <no. samples> <dt between samples> <path to file with data>\n" );

    size_t degree;
    early_exit_test( parse_ulong(argv[1], degree), "Error: <degree> is not a number\n" );

    early_exit_test( MIN_DEGREE <= degree && degree <= MAX_DEGREE, "Error: degree is too large or too small. Compile to a higher degree.\n" );

    size_t no_samples;
    early_exit_test( parse_ulong(argv[2], no_samples), "Error: <no. samples> is not a number\n" );

    double dt;
    early_exit_test( parse_double(argv[3], dt), "Error: <dt between samples> is not a number\n" );

    const char* path = argv[4];

    int file_fd = open(path, O_RDONLY);
    early_exit_test(file_fd != -1, "Failed to open file <path to file with data>\n");

    size_t num_floats = 3 * no_samples;

    std::unique_ptr<float[]> data_owner = std::make_unique<float[]>(num_floats);
    ssize_t amt_read = read(file_fd, data_owner.get(), sizeof(float) * num_floats);

    early_exit_test( amt_read == sizeof(float) * num_floats, "Error in reading no. samples amount of data.\n" );
    
    std::span<float> data_view( data_owner.get(), num_floats);

    for ( auto f : data_view ) std::cout << f << ' ';
    std::cout << '\n';

    magnus(degree, no_samples, dt, nullptr, nullptr);
    
}