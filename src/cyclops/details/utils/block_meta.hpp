#pragma once

#include <array>

namespace cyclops::block_meta {
  struct position {
    static std::size_t constexpr size = 3;
  };

  struct velocity {
    static std::size_t constexpr size = 3;
  };

  struct orientation {
    static std::size_t constexpr size = 4;
  };

  struct bias_acc {
    static std::size_t constexpr size = 3;
  };

  struct bias_gyr {
    static std::size_t constexpr size = 3;
  };

  struct landmark_position {
    static std::size_t constexpr size = 3;
  };

  template <typename meta, typename... rest>
  struct meta_size_sum {
    static std::size_t constexpr result =
      meta::size + meta_size_sum<rest...>::result;
  };

  template <typename meta>
  struct meta_size_sum<meta> {
    static std::size_t constexpr result = meta::size;
  };

  template <typename... metas>
  using block_cascade = std::array<double, meta_size_sum<metas...>::result>;
}  // namespace cyclops::block_meta
