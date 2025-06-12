#include "cyclops/details/estimation/ceres/cost.imu_preintegration.hpp"

namespace cyclops::estimation {
  using Matrix9d = Eigen::Matrix<double, 9, 9>;

  static Matrix9d makeImuPreintegrationResidualWeight(
    measurement::ImuPreintegration const* data) {
    return Eigen::LLT<Matrix9d>(data->covariance.inverse()).matrixU();
  }

  ImuPreintegrationCostEvaluator::ImuPreintegrationCostEvaluator(
    measurement::ImuPreintegration const* data, double gravity)
      : data(data),
        weight(makeImuPreintegrationResidualWeight(data)),
        gravity(gravity) {
  }
}  // namespace cyclops::estimation
