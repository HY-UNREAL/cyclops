target_sources(cyclops
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/acceptance.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/acceptance.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/acceptance.imu_only.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/acceptance.imu_only.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/analysis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/analysis.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/analysis_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/analysis_cache.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hessian.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hessian.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/motion_prior.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/motion_prior.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_evaluation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_evaluation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_refinement.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_refinement.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_sample.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_sample.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solver.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solver.imu_only.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solver.imu_only.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/uncertainty.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/uncertainty.hpp
)

cyclops_add_unittest(
  initializer_motion_prior
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/motion_prior.test.cpp
)

cyclops_add_unittest(
  initializer_solver
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/solver.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/acceptance.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/analysis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/analysis_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hessian.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_evaluation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_refinement.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_sample.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/uncertainty.cpp
  ${cyclops_src_dir}/cyclops/details/measurement/preintegration.cpp
  ${cyclops_src_dir}/cyclops/details/telemetry/initializer.cpp
  ${cyclops_src_dir}/cyclops/details/utils/debug.cpp
  ${cyclops_src_dir}/cyclops/details/utils/math.cpp
  ${cyclops_src_dir}/cyclops/details/utils/qcqp1.cpp

  DEPENDS cyclops_testlib
)

cyclops_add_unittest(
  initializer_solver_imu_only
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/solver.imu_only.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/acceptance.imu_only.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/analysis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/analysis_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hessian.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_evaluation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/uncertainty.cpp
  ${cyclops_src_dir}/cyclops/details/measurement/preintegration.cpp
  ${cyclops_src_dir}/cyclops/details/utils/debug.cpp
  ${cyclops_src_dir}/cyclops/details/utils/math.cpp
  ${cyclops_src_dir}/cyclops/details/utils/qcqp1.cpp

  DEPENDS cyclops_testlib
)


cyclops_add_unittest(
  initializer_analysis
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/analysis.test.cpp
  ${cyclops_src_dir}/cyclops/details/measurement/preintegration.cpp
)

cyclops_add_unittest(
  initializer_translation_primal
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/scale_evaluation.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/analysis_cache.cpp
  ${cyclops_src_dir}/cyclops/details/utils/qcqp1.cpp
)
