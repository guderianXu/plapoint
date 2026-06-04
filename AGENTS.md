# AGENTS.md

本文件给后续 agent 使用，优先级低于用户当前指令和系统/开发者指令。目标是在 PlaPoint 项目里改代码、跑测试、维护文档和同步子模块时保持一致做法。

## 项目定位

PlaPoint 是基于 PlaMatrix 的 C++17 点云处理库，支持 CPU 实现和可选 CUDA 加速。主线包含：

- `include/plapoint/core/`：`PointCloud<Scalar, Device>`、点属性、CPU/GPU 数据传输。
- `include/plapoint/search/`：CPU kd-tree、GPU brute-force KNN 和批量 KNN。
- `include/plapoint/filters/`：体素下采样、统计离群点、半径离群点、均匀下采样。
- `include/plapoint/features/`：法线估计、法线平滑和视点定向。
- `include/plapoint/registration/`：点到点 ICP。
- `include/plapoint/mesh/`：Marching Cubes 和 Poisson 重建。
- `include/plapoint/io/`：PLY、OBJ、XYZ、LAS 等点云 I/O。
- `src/*.cu`：CUDA KNN、批量 KNN、voxel grid kernels。
- `test/`：Google Test 单元测试和 `test/validation/reference_data/` 参考数据。

先读现有实现、测试和 README，再改动。优先延续当前模块边界和命名风格，避免无关重构。

## 工作区约束

- 本目录既是独立 Git 仓库，也是上层 `plascan/3rdparty` 下的子模块。远端仓库是 `https://github.com/guderianXu/plapoint.git`，当前主分支为 `master`。
- 开始改动前检查：
  ```bash
  git status --short
  git branch --show-current
  ```
- 如果在上层项目中工作，只有在用户明确要求时才更新上层仓库的子模块指针。
- 不要删除或批量重写 `test/validation/reference_data/`。只有算法预期值确实变化时，才重新生成并同时说明命令、依赖和差异原因。
- PlaPoint 依赖 `plamatrix::plamatrix`。不要为了临时绕过问题复制 PlaMatrix 代码；确实需要改底层矩阵行为时，在 `plamatrix` 仓库中单独处理。
- 不要把 GitHub token、私有路径、临时 PID 或本机凭据写进文档、脚本或 remote URL。

## 代码规范

### C++

- 使用 C++17，4 空格缩进，不使用 tab，行宽尽量不超过 120 字符。
- 新增或大幅重写的代码使用 Allman brace style，左大括号单独成行。编辑旧代码时优先保持邻近风格，不做纯格式化 churn。
- 类和结构体使用 PascalCase，例如 `PointCloud`、`VoxelGrid`。
- 函数和方法使用 camelCase，例如 `nearestKSearch()`、`setInputCloud()`。
- 局部变量和普通成员变量使用 snake_case，例如 `point_count`、`leaf_size`。
- 私有成员变量前加下划线，例如 `_cloud`、`_nodes`。
- 常量和宏使用大写下划线，例如 `PLAPOINT_WITH_CUDA`。
- 命名空间使用 `plapoint` 及其子命名空间，例如 `plapoint::search`、`plapoint::gpu`。
- 头文件使用 `.h` 和 `#pragma once`；CUDA 源文件使用 `.cu`；纯 C++ 源文件使用 `.cpp`。
- 公共声明需要简短文档注释，说明参数、返回值和重要异常条件。
- 新文件 include 顺序优先为标准库、CUDA/第三方、PlaMatrix、PlaPoint；编辑旧文件时不要只为 include 排序制造无关 diff。
- CPU 侧参数非法、维度不匹配、文件解析失败和构造失败优先抛出明确异常。
- GPU 侧 CUDA 调用使用 `PLAPOINT_CHECK_CUDA` 或现有封装，不留下未检查的 CUDA runtime 调用。
- 一个函数只承担清晰职责。文件超过 400 行或嵌套超过 4 层时优先拆小，但不要为了机械满足限制做无意义拆分。

### 点云和数值约定

- `PointCloud` 的点矩阵必须是 Nx3。normals、colors、texture coords、faces 等可选属性必须校验行数和列数，不要静默截断。
- PlaMatrix 使用列优先矩阵。需要给 CUDA kernel 使用行优先临时缓冲时，转换逻辑必须显式、可测试，并说明数据布局。
- CPU 侧点访问优先使用 `operator()`；跨设备通用路径使用 `getValue()` / `setValue()`，注意 GPU 单元素拷贝成本。
- KNN GPU 路径当前支持 `k <= 32` 且 `k <= point_count`。扩大限制时必须同时改 kernel、测试和错误信息。
- 过滤器和重建算法要保持确定性。涉及浮点容差时，在测试中写明理由，不要随意放宽断言。
- I/O 代码必须保留文件声明顺序和可选属性；新增格式支持时至少覆盖正常文件、空文件、缺失字段和非法字段。

## 构建与测试

常用开发验证：

```bash
cmake -S . -B build -DPLAPOINT_BUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/test/plapoint_tests
ctest --test-dir build --output-on-failure
```

CPU-only 验证：

```bash
cmake -S . -B build-cpu -DPLAPOINT_BUILD_TESTS=ON -DPLAPOINT_WITH_CUDA=OFF
cmake --build build-cpu -j$(nproc)
ctest --test-dir build-cpu --output-on-failure
```

如果独立构建找不到 PlaMatrix，先安装或指定前缀：

```bash
cmake -S ../plamatrix -B ../plamatrix/build -DPLAMATRIX_BUILD_TESTS=ON
cmake --build ../plamatrix/build -j$(nproc)
cmake --install ../plamatrix/build --prefix ../plamatrix/install
cmake -S . -B build -DPLAPOINT_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=../plamatrix/install
```

改动选择验证范围：

- 改核心 `PointCloud` 或属性：跑 `point_cloud_*`、I/O 和全量 `ctest`。
- 改 KNN/kd-tree：跑 `kdtree_test`、`kdtree_gpu_test`、validation 测试和 CPU-only 构建。
- 改 filters/features/registration/mesh：跑对应模块测试，并确认参考数据是否仍匹配。
- 改 CUDA kernel：至少跑 CUDA 构建、CPU-only 构建和相关 GPU 测试。
- 改 I/O：覆盖 ASCII/binary、可选属性、非法输入和 round-trip。

重新生成参考数据前先确认 Python 依赖和变更原因：

```bash
python3 test/validation/generate_reference.py
```

## Git 与同步

- 如用户要求提交，先检查 `git status --short`，保留用户未授权的改动。
- 提交作者建议使用 GitHub 关联身份：
  ```bash
  git config user.name "guderianXu"
  git config user.email "guderian_xu@henu.edu.cn"
  ```
- 不要在未经用户要求时 commit、push、改 remote 或更新上层子模块指针。
- 若用户要求发布到 GitHub，使用不含凭据的 remote URL：
  ```bash
  git remote set-url origin https://github.com/guderianXu/plapoint.git
  ```

## 文档与沟通

- 对用户用中文简洁说明做了什么、验证了什么、还有什么风险。
- API、构建选项、CUDA 行为、I/O 格式或依赖方式变化时，同步更新 `README.md` 和相关示例。
- 涉及性能、显存、点数规模和测试结果的结论要给出具体命令、硬件条件或测试路径。
- 面向人工留档的报告优先写 Markdown；机器消费的数据才写 JSON/CSV/bin/npy。
