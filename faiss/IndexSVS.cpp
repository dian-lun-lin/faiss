/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVS.h>
#include "faiss/Index.h"

namespace faiss {

IndexSVS::IndexSVS(
    idx_t d, 
    MetricType metric,
    size_t num_threads,
    size_t graph_max_degree
):Index(d, metric), num_threads{num_threads}, graph_max_degree{graph_max_degree} {
}

void IndexSVS::add(idx_t n, const float* x) {

    if(!impl) {
        init_impl(n, x);
        return;
    }

    // construct sequential labels
    std::vector<size_t> labels(n);

    svs::threads::parallel_for(
        impl->get_threadpool_handle(),
        svs::threads::StaticPartition(n),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for(auto i : is) {
                labels[i] = ntotal + i;
            }
        }
    );
    ntotal += n;

    auto data = svs::data::ConstSimpleDataView<float>(x, n, d);
    impl->add_points(data, labels);
}

void IndexSVS::reset() {
    if(impl) {
        delete impl;
    }
    ntotal = 0;
}

IndexSVS::~IndexSVS() {
    if(impl) {
        delete impl;
    }
}

void IndexSVS::search(
    idx_t n,
    const float* x,
    idx_t k,
    float* distances,
    idx_t* labels,
    const SearchParameters* params
) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    auto queries = svs::data::ConstSimpleDataView<float>(x, n, d);

    // TODO: use params for SVS search parameters
    auto sp = impl->get_search_parameters();
    sp.buffer_config({search_window_size, search_buffer_capacity});
    // TODO: faiss use int64_t as label whereas SVS uses size_t?
    auto results = svs::QueryResultView<size_t>{svs::MatrixView<size_t>{svs::make_dims(n, k), static_cast<size_t*>(static_cast<void*>(labels))}, svs::MatrixView<float>{svs::make_dims(n, k), distances}};
    impl->search(results, queries, sp);
}

void IndexSVS::init_impl(idx_t n, const float* x) {
    std::vector<size_t> labels(n);
    auto data = svs::data::SimpleData<float>(n, d);
    auto threadpool = svs::threads::as_threadpool(num_threads);

    svs::threads::parallel_for(
        threadpool,
        svs::threads::StaticPartition(n),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for(auto i : is) {
                data.set_datum(i, std::span<const float>(x + i * d, d));
                labels[i] = ntotal + i;
            }
        }
    );
    ntotal += n;

    svs::index::vamana::VamanaBuildParameters build_parameters{alpha, graph_max_degree, construction_window_size, max_candidate_pool_size, prune_to, use_full_search_history};

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
          impl = new svs::DynamicVamana(svs::DynamicVamana::build<float>(
                std::move(build_parameters),
                std::move(data), 
                std::move(labels),
                svs::DistanceIP(),
                std::move(threadpool))
              );
          break;
        case METRIC_L2:
          impl = new svs::DynamicVamana(svs::DynamicVamana::build<float>(
                std::move(build_parameters),
                std::move(data),
                std::move(labels),
                svs::DistanceL2(),
                std::move(threadpool))
              );
          break;
        default:
          FAISS_ASSERT(!"not supported SVS distance");
    }
}

} // namespace faiss
