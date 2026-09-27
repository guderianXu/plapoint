#include <gtest/gtest.h>

#if defined(PLAPOINT_WITH_CUDA) && defined(PLAPOINT_REQUIRE_CUDA_TEST_DEVICE)

#include <cuda_runtime.h>

TEST(CudaRuntimeRequirementTest, HasUsableDevice)
{
    int device_count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    ASSERT_EQ(count_error, cudaSuccess) << "CUDA runtime device discovery failed: " << cudaGetErrorString(count_error);
    ASSERT_GT(device_count, 0) << "CUDA runtime validation requires at least one visible device";

    const cudaError_t context_error = cudaFree(nullptr);
    ASSERT_EQ(context_error, cudaSuccess)
        << "CUDA context initialization failed: " << cudaGetErrorString(context_error);
}

#endif
