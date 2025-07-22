/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include <numeric>

#include "faiss/Index.h"
#include <faiss/impl/FaissAssert.h>

#include "svs/orchestrators/vamana.h"
#include "svs/orchestrators/dynamic_vamana.h"

namespace faiss {

struct IndexSVS : Index {

  IndexSVS(
      idx_t d, 
      MetricType metric = METRIC_L2,
      size_t num_threads = 1,
      size_t graph_max_degree = 64
  );
  // rn OMP_NUM_THREADS

  ~IndexSVS() override;

  void add(idx_t n, const float* x) override;

  void search(
      idx_t n,
      const float* x,
      idx_t k,
      float* distances,
      idx_t* labels,
      const SearchParameters* params = nullptr) const override;

  void reset();

  virtual void init_impl(idx_t n, const float* x);

  svs::DynamicVamana* impl{nullptr};

  size_t num_threads;
  size_t graph_max_degree;
  // default parameters
  float alpha = 1.2;
  size_t search_window_size = 10;
  size_t search_buffer_capacity = 10;
  size_t construction_window_size = 40;
  size_t max_candidate_pool_size = 200;
  size_t prune_to = 60;
  bool use_full_search_history = true;
};

} // namespace faiss
