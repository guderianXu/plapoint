#include <plapoint/opencl/opencl_runtime.h>

int main()
{
    return plapoint::opencl::hasUsableOpenClDevice() ? 1 : 0;
}
