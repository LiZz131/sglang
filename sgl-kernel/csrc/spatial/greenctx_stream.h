#include <vector>
std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device);
std::vector<int64_t> create_greenctx_streams_by_value_enhanced(
    int64_t smA,
    int64_t smB,
    int64_t n_streams_a,
    int64_t n_streams_b,
    int64_t device);
