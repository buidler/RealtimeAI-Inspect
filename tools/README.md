### 入门指南：完整工作流程

本指南提供了一个完整的逐步工作流程，从环境设置到训练、导出以及使用 TensorRT 进行推理。

#### **1. 使用 Docker 配置环境（推荐）**

使用 Docker 是推荐的方式，可以确保所有依赖项、驱动程序和 CUDA 版本完美匹配。这消除了“在我的机器上可以运行”的问题。

*   **步骤 1.1: 构建并运行容器**

    从项目根目录运行 `docker compose`。这将基于 `Dockerfile` 构建镜像并在后台启动服务。

    ```bash
    docker compose up --build -d
    ```

*   **步骤 1.2: 验证容器是否正在运行**

    检查容器是否已启动并正在运行。记下其名称以备下一步使用。
    ```bash
    docker ps
    ```

---

#### **2. 训练与评估（使用 `docker attach`）**

此方法直接将您的终端连接到容器的主进程。它很简单，但需要小心操作以避免意外终止会话。

*   **步骤 2.1: 连接到容器**

    将您的终端连接到正在运行的容器。您将进入一个 bash shell。

    ```bash
    docker attach <your_container_name>
    ```

*   **步骤 2.2: 运行训练命令**

    现在，在*连接的 shell 内部*，运行您的训练命令。`torchrun` 将自动使用分配给容器的 GPU。**不要在后台运行它（即不要加 `&`）**。

    ```bash
    # 示例：分配给容器 4 个 GPU
    torchrun --nproc_per_node=4 --master-port=8989 \
        tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
        --amp
    ```

*   **步骤 2.3: 从会话中断开（重要！）**

    当您的训练正在运行时，您可以安全地断开连接并让它继续运行。

    **警告：** **不要按 `Ctrl+C`**。这将终止训练进程，并可能终止整个容器。

    要安全断开连接，请按顺序按下：**`Ctrl+P`**，然后立即按 **`Ctrl+Q`**。

    您将返回到本地终端，容器将在后台继续运行训练。

*   **步骤 2.4: 重新连接到您的会话**

    要检查训练进度，只需再次运行 `docker attach` 命令。您将看到训练命令的实时输出。

    ```bash
    docker attach <your_container_name>
    ```
    （请记住，完成后使用 `Ctrl+P`, `Ctrl+Q` 断开连接。）

---

#### **3. 导出与推理**

对于导出或运行推理等不需要运行几天的任务，使用 `docker exec` 打开一个新的、独立的 shell 更安全。

*   **步骤 3.1: 在容器中打开一个新 Shell**
    ```bash
    docker exec -it <your_container_name> bash
    ```

*   **步骤 3.2: 运行导出或推理命令**
    现在，在这个新 shell 中，运行您的命令。
    ```bash
    # 导出为 ONNX
    python tools/export_onnx.py \
        -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
        -r path/to/trained_checkpoint.pth \
        --check
    ```
    
    ```
    # 转换为 TensorRT
    bash tools/onnx2trt.sh /path/to/your/model.onnx
    ```

    ```
    # 运行 TRT 推理
    python references/deploy/rtdetrv2_tensorrt.py \
    --engine /path/to/your/model.trt \
    --image /path/to/your/image.jpg \
    --output /path/to/save/output.jpg \
    --threshold 0.5
    ```

### 实用工具与技巧

*   **使用 TensorBoard 可视化训练：**
    *   使用标准端口 `6006` 以避免与训练冲突。
    *   确保端口 `6006` 在您的 `docker-compose.yml` 中已暴露。

    ```bash
    # 在容器内部
    tensorboard --logdir=path/to/summary/ --host=0.0.0.0 --port=6006
    ```

*   **管理容器生命周期：**
    *   **临时停止**容器而不删除它（例如，暂停训练并在稍后恢复）：
        ```bash
        docker compose stop
        ```
        您可以稍后使用 `docker compose start` 重启它。

    *   **停止并完全移除**容器、网络和卷：
        ```bash
        docker compose down
        ```
