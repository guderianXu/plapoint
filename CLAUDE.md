# CLAUDE.md
# 通用
- 优先选择编辑而非重写整个文件
- 除非文件被编辑过，否则不要重复阅读已读过的文件
- 输出追求简洁，但推理过程必须详尽

# 代码规范

## 命名规范
- **类/结构体**：大驼峰（PascalCase），如 `DenseMatrix`, `CudaAllocator`
- **函数/方法**：小驼峰（camelCase），如 `toCpu()`, `solveLinear()`
- **变量**：全小写下划线分隔（snake_case），如 `row_count`, `data_ptr`
- **私有成员变量**：首字母前加下划线，如 `_data`, `_rows`
- **常量/宏**：全大写+下划线，如 `MAX_THREADS`, `PLAMATRIX_VERSION`
- **模板参数**：首字母大写驼峰或单大写字母，如 `Scalar`, `Device`, `N`
- **命名空间**：全小写单单词，如 `plamatrix`

## 格式规范
- 4 空格缩进，不使用 tab
- 
- 行宽上限 120 字符
- 头文件函数声明必须写文档注释（`///` 或 `/** */`），说明参数、返回值、异常条件

## 文件组织
- 一个文件不超过 400 行，超了就拆
- 嵌套不超过 4 层
- 头文件使用 `.h` 扩展名，CUDA 源文件使用 `.cu` 扩展名，纯 C++ 使用 `.cpp`
- 头文件使用 `#pragma once`

## 错误处理
- **CPU 侧**：使用异常（`throw std::runtime_error` 等），构造失败、参数非法时抛出
- **GPU 侧**：kernel 内使用错误码，host 侧 CUDA 调用用 `PLAMATRIX_CHECK_CUDA(call)` 宏封装，失败抛异常
- 断言：运行时参数校验用 `assert()` 或自定义 `PLAMATRIX_ASSERT`

## include 顺序
1. 标准库头文件
2. CUDA 头文件（`cuda_runtime.h`, `cublas_v2.h`, `cusolverDn.h`）
3. 第三方库头文件
4. 项目头文件（`plamatrix/...`）
每组之间空行分隔，每组内按字母序

## 测试规范
- 测试文件命名：`*_test.cpp`
- 测试用例命名：`ClassName_MethodName_Scenario`
- 测试框架：Google Test + CUDA 测试宏

# Git 同步
- **每次 commit 后必须 push 到 GitHub**：`git push origin main`（含 tags: `git push origin <tag>`）
- 仓库地址：`https://github.com/guderianXu/plapoint`
- **Git 作者配置**：必须使用 GitHub 关联邮箱，否则提交不计入贡献统计
  - `git config user.email "guderian_xu@henu.edu.cn"`
  - `git config user.name "guderianXu"`
  - 注意：`guderian@plapoint.local` 是不关联 GitHub 的本地邮箱，不要使用
  - 如已用错误邮箱提交，需要用 `git filter-branch --env-filter` 重写历史

# 编译验证
- **每次修改代码后必须在 build 目录编译验证**：
  ```bash
  cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc)
  ```
- 编译失败不能提交；编译警告需评估

# 开发流程 (Feature Branch + TDD)

**分支策略**：
- `main` 分支始终稳定可构建，不直接在 main 上开发
- 每个功能/修复在独立 feature 分支上开发：`feat/<描述>` 或 `fix/<描述>`
- 分支上完成 TDD 循环（红→绿→重构）且测试全部通过后，合并回 main

**TDD 铁律**：
1. 先写测试 → 运行确认失败（红）
2. 写最小实现 → 运行确认通过（绿）
3. 重构优化 → 保持测试通过
4. 每次提交前跑相关测试，禁止提交破坏测试的代码

**分支操作规范**：
```bash
git checkout -b feat/<功能名>   # 从 main 创建功能分支
# ... TDD 开发 ...
git checkout main && git merge feat/<功能名>   # 测试通过后合并
git push origin main
```

