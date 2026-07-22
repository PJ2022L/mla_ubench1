# H800 Docker 运行环境

本文说明如何在远端 H800 宿主机启动容器并运行本项目。仓库在远端的实际路径不固定；宿主机先进入仓库根目录，再通过 `pwd` 得到挂载源。容器内不使用 conda，直接使用镜像中的 `python`。

## 1. 前置条件

宿主机需要安装：

- NVIDIA Driver，且能够识别 H800；
- Docker；
- NVIDIA Container Toolkit。

先在宿主机验证：

```bash
nvidia-smi

docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu22.04 \
  nvidia-smi
```

第二条命令必须能在容器中看到 H800。失败时先修复宿主机的 NVIDIA Container Toolkit，不要继续运行 benchmark。

## 2. 镜像要求

镜像至少需要：

- CUDA 12.8 或更高版本；
- CUDA development toolkit 和 `nvcc`，不能只用 runtime 镜像；
- 支持 CUDA 的 PyTorch；
- C++/CUDA extension 构建工具。

下面使用 `nvcr.io/nvidia/pytorch:25.04-py3` 作为示例。若远端已有内部镜像，可替换 `IMAGE`，但必须先确认上述条件。

拉取 NGC 镜像前可能需要登录：

```bash
docker login nvcr.io
```

## 3. 创建容器

以下命令全部在 H800 宿主机执行。先进入仓库根目录：

```bash
repo_root="$(pwd)"
test -f "$repo_root/HANDOFF.md"
test -f "$repo_root/AGENT.md"

IMAGE=nvcr.io/nvidia/pytorch:25.04-py3

docker run -it \
  --name mla-h800 \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$repo_root:/workspace/mla_ubench1" \
  -w /workspace/mla_ubench1 \
  "$IMAGE" \
  bash
```

关键参数：

- `--gpus all`：向容器暴露 GPU；只使用第 0 张卡时可改为 `--gpus device=0`。
- `--ipc=host` 和 `--shm-size=32g`：为 PyTorch 和较大 tensor 提供共享内存。
- `SYS_ADMIN`：允许 Nsight Compute 访问 GPU performance counter；宿主机驱动策略仍可能额外限制 profiling。
- `SYS_PTRACE` 和 `seccomp=unconfined`：支持 `ncu`/`nsys` 等 profiler。
- `-v`：把当前远端仓库挂载进容器，代码和 `result/` 会持久化到宿主机。
- `-w`：容器启动后位于仓库根目录。

不要添加 `--rm`。保留命名容器后，安装在容器文件系统中的 Python 依赖可在后续实验中继续使用。

## 4. 容器内检查

进入容器后执行：

```bash
repo_root="$(pwd)"
test -f "$repo_root/HANDOFF.md"

nvidia-smi -L
nvcc --version

python -c '
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
'
```

H800 的 CUDA capability 应为 `(9, 0)`。`nvcc` 必须存在，且 CUDA 版本满足 FlashMLA 的要求。

## 5. 安装 FlashMLA

仍在容器的仓库根目录执行：

```bash
repo_root="$(pwd)"

git -C "$repo_root/operators/flash_mla/target" \
  submodule update --init --recursive

FLASH_MLA_DISABLE_SM100=1 \
python -m pip install -v \
  "$repo_root/operators/flash_mla/target"

python -c 'import flash_mla; print(flash_mla)'
```

安装失败时保存完整构建日志，并检查 PyTorch CUDA、`nvcc`、host driver 和 CUTLASS submodule 版本。不要通过降低目标架构绕过 SM90a 编译错误。

## 6. 运行 micro-benchmark

先完成静态门禁，再按 [HANDOFF.md](HANDOFF.md) 的任务顺序执行：

```bash
make -C microbench validate
make -C microbench dry-build
make -C microbench build
make -C microbench static

make -C operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/calibration check
make -C operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/calibration build

python3 microbench/scripts/run_all.py --mode quick --device 0
python3 microbench/scripts/run_all.py --mode full --device 0
```

每个 family 的正式输出是其 `result.csv`。只有完整成功的 full sweep 会原子替换
该文件；完整 args、命令、wall time、失败信息和原始六键 JSON 保存在同一 family 的
`build/logs/` 与 `build/raw/`。

## 7. 运行 dense e2e

```bash
repo_root="$(pwd)"
e2e_dir="$repo_root/operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/e2e"

run_id="$(date +%Y%m%d-%H%M%S)-$(hostname)"
run_dir="$e2e_dir/result/runs/$run_id"
mkdir -p "$run_dir"

/usr/bin/time -f 'command=%C\nwall_seconds=%e\nexit_status=%x' \
  python "$e2e_dir/benchmark.py" \
    --batch 128 --s-q 1 --s-k 4096 --warmup 10 --iters 100 \
  >"$run_dir/results.jsonl" 2>"$run_dir/run.log"
```

CUDA event 测量由 `benchmark.py` 完成；`run.log` 只记录命令、wall time 和错误，
不参与 GPU 计时。

## 8. 重新进入和停止容器

退出后重新进入同一个容器：

```bash
docker start -ai mla-h800
```

在另一个终端进入正在运行的容器：

```bash
docker exec -it mla-h800 bash
```

停止容器：

```bash
docker stop mla-h800
```

确认不再需要容器内安装环境后才删除：

```bash
docker rm mla-h800
```

删除容器不会删除 bind-mounted 仓库及其 `result/`，但会删除只安装在容器文件系统中的依赖。

## 9. Profiler 权限问题

如果 `ncu` 报 `ERR_NVGPUCTRPERM`，仅修改容器参数可能不够。需要宿主机管理员允许访问 NVIDIA performance counter，或按集群策略以具有相应权限的方式启动容器。不要在没有 profiler 证据时声称已经确认 main/combine、opcode 或硬件瓶颈。
