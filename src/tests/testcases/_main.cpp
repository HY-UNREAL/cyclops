#include "cyclops/details/logging.hpp"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

int main(int argc, char* argv[]) {
  doctest::Context context(argc, argv);
  cyclops::initLogger(1);

  return context.run();
}
