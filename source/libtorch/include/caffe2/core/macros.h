// Automatically generated header file for caffe2 macros. These
// macros are used to build the Caffe2 binary, and if you are
// building a dependent library, they will need to be set as well
// for your program to link correctly.

#pragma once

// Caffe2 version. The plan is to increment the minor version every other week
// as a track point for bugs, until we find a proper versioning cycle.

#define CAFFE2_VERSION_MAJOR 1
#define CAFFE2_VERSION_MINOR 10
#define CAFFE2_VERSION_PATCH 2

static_assert(
    CAFFE2_VERSION_MINOR < 100,
    "Programming error: you set a minor version that is too big.");
static_assert(
    CAFFE2_VERSION_PATCH < 100,
    "Programming error: you set a patch version that is too big.");

#define CAFFE2_VERSION                                         \
  (CAFFE2_VERSION_MAJOR * 10000 + CAFFE2_VERSION_MINOR * 100 + \
   CAFFE2_VERSION_PATCH)

#define CAFFE2_BUILD_SHARED_LIBS
/* #undef CAFFE2_FORCE_FALLBACK_CUDA_MPI */
/* #undef CAFFE2_HAS_MKL_DNN */
/* #undef CAFFE2_HAS_MKL_SGEMM_PACK */
#define CAFFE2_PERF_WITH_AVX
#define CAFFE2_PERF_WITH_AVX2
#define CAFFE2_PERF_WITH_AVX512
/* #undef CAFFE2_THREADPOOL_MAIN_IMBALANCE */
/* #undef CAFFE2_THREADPOOL_STATS */
#define CAFFE2_USE_EXCEPTION_PTR
/* #undef CAFFE2_USE_ACCELERATE */
/* #undef CAFFE2_USE_CUDNN */
/* #undef CAFFE2_USE_EIGEN_FOR_BLAS */
/* #undef CAFFE2_USE_FBCODE */
/* #undef CAFFE2_USE_GOOGLE_GLOG */
/* #undef CAFFE2_USE_LITE_PROTO */
#define CAFFE2_USE_MKL
#define CAFFE2_USE_MKLDNN
/* #undef CAFFE2_USE_NVTX */
/* #undef CAFFE2_USE_TRT */

#ifndef EIGEN_MPL2_ONLY
#define EIGEN_MPL2_ONLY
#endif

// Useful build settings that are recorded in the compiled binary
#define CAFFE2_BUILD_STRINGS { \
  {"TORCH_VERSION", "1.10.2"}, \
  {"CXX_COMPILER", "/Applications/Xcode-12.0.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++"}, \
  {"CXX_FLAGS", " -Wno-deprecated -fvisibility-inlines-hidden -Wno-deprecated-declarations -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-typedef-redefinition -Wno-unknown-warning-option -Wno-unused-private-field -Wno-inconsistent-missing-override -Wno-aligned-allocation-unavailable -Wno-c++14-extensions -Wno-constexpr-not-const -Wno-missing-braces -Qunused-arguments -fcolor-diagnostics -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-unused-private-field -Wno-missing-braces -Wno-c++14-extensions -Wno-constexpr-not-const"}, \
  {"BUILD_TYPE", "Release"}, \
  {"BLAS_INFO", "mkl"}, \
  {"LAPACK_INFO", "mkl"}, \
  {"USE_CUDA", "OFF"}, \
  {"CUDA_VERSION", ""}, \
  {"USE_CUDNN", "OFF"}, \
  {"CUDNN_VERSION", ""}, \
  {"USE_NCCL", "OFF"}, \
  {"USE_MPI", "OFF"}, \
  {"USE_GFLAGS", "OFF"}, \
  {"USE_GLOG", "OFF"}, \
  {"USE_GLOO", ""}, \
  {"USE_NNPACK", "ON"}, \
  {"USE_OPENMP", "OFF"}, \
  {"FORCE_FALLBACK_CUDA_MPI", ""}, \
  {"HAS_MKL_DNN", ""}, \
  {"HAS_MKL_SGEMM_PACK", ""}, \
  {"PERF_WITH_AVX", "1"}, \
  {"PERF_WITH_AVX2", "1"}, \
  {"PERF_WITH_AVX512", "1"}, \
  {"USE_EXCEPTION_PTR", "1"}, \
  {"USE_ACCELERATE", ""}, \
  {"USE_EIGEN_FOR_BLAS", ""}, \
  {"USE_LITE_PROTO", ""}, \
  {"USE_MKL", "ON"}, \
  {"USE_MKLDNN", "ON"}, \
  {"USE_NVTX", ""}, \
  {"USE_TRT", ""}, \
}
