#pragma once

#include <Eigen/Dense>

#include <memory>

namespace cyclops::initializer {
  struct ImuMatchAnalysis;

  class ImuMatchAnalysisCache {
  private:
    struct Impl;
    std::unique_ptr<Impl> _pimpl;

  public:
    explicit ImuMatchAnalysisCache(ImuMatchAnalysis const& analysis);
    ~ImuMatchAnalysisCache();

    /*
     * Represents the following Schur-decomposition of linear system of
     * equations,
     *
     *  [H_I + mu * C_I,         F_I * s] * [ x_I ] + [ b_I(s) ]  = [ 0 ]
     *  [F_I^T * s,      H_V + D_I * s^2]   [ x_V ]   [ b_V(s) ]    [ 0 ]
     *
     *                                <=>
     *
     *            (H_I_bar + mu * C_I) * x_I + b_I_bar = 0
     *                  x_V = -(F_V * x_I + z_V) * s
     * .
     */
    struct PrimalCacheInflation {
      Eigen::MatrixXd H_I_bar;
      Eigen::VectorXd b_I_bar;
      Eigen::MatrixXd F_V;
      Eigen::VectorXd z_V;
    };
    PrimalCacheInflation inflatePrimal(double scale) const;

    struct DerivativeCacheInflation {
      double r_s__dot;
      Eigen::VectorXd b_I_s__dot;
      Eigen::VectorXd b_V_s__dot;
      Eigen::MatrixXd F_I;
      Eigen::MatrixXd D_I;
    };
    DerivativeCacheInflation inflateDerivative(double scale) const;
  };
}  // namespace cyclops::initializer
