#pragma once

#include <range/v3/all.hpp>

#include <vector>
#include <set>

namespace cyclops {
  template <typename value_t>
  class DisjointSetPartitionContext {
  private:
    std::vector<std::set<value_t>> _sets;
    std::vector<int> _parents;

    bool intersects(std::set<value_t> const& s1, std::set<value_t> const& s2) {
      return !ranges::empty(ranges::views::set_intersection(s1, s2));
    }

    int findRoot(int i) {
      if (_parents.at(i) == i)  // No parents; i.e. root.
        return i;
      return _parents.at(i) = findRoot(_parents.at(i));
    }

    void unite(int i, int j) {
      auto root_i = findRoot(i);
      auto root_j = findRoot(j);

      if (root_i != root_j)
        _parents.at(root_i) = root_j;
    };

  public:
    explicit DisjointSetPartitionContext(
      std::vector<std::set<value_t>> const& sets)
        : _sets(sets) {
      if (sets.size() <= 1)
        return;

      auto n = static_cast<int>(sets.size());
      _parents = ranges::views::ints(0, n) | ranges::to_vector;
    }

    std::vector<std::set<value_t>> operator()() {
      auto n = static_cast<int>(_sets.size());

      for (auto i = 0; i < n; i++) {
        for (auto j = i + 1; j < n; j++) {
          if (intersects(_sets.at(i), _sets.at(j)))
            unite(i, j);
        }
      }

      std::map<int, std::set<value_t>> merged_sets;
      for (auto i = 0; i < n; i++) {
        auto const& set = _sets.at(i);
        merged_sets[findRoot(i)].insert(set.begin(), set.end());
      }

      return merged_sets | ranges::views::values | ranges::to_vector;
    }
  };
}  // namespace cyclops
