# FactorMiner 项目架构与算法关系图

本文档提供了 FactorMiner 的整体系统架构图和核心算法流程的关系图。所有图表均使用 Mermaid 语法生成。

## 1. 系统整体架构图

该图展示了 FactorMiner 中数据、核心循环、内存策略、评估以及分析/基准测试模块之间的总体依赖关系。

```mermaid
flowchart TD
    subgraph Input ["输入层"]
        A["市场数据 (Market Data)"]
    end

    subgraph Core ["架构与核心契约 (Architecture & Contracts)"]
        B["DatasetContract"]
        C["强类型领域特定语言 (Typed DSL)\n+ 操作符注册表 (Operator Registry)"]
    end

    subgraph Execution ["执行通道 (Execution Lanes)"]
        D["Ralph 循环 / Helix 循环\n(Pipeline Stages)"]
    end

    subgraph Algorithm ["算法与评估 (Algorithms & Evaluation)"]
        E["评估内核 (EvaluationKernel)"]
        F["因子准入服务 (FactorAdmissionService)"]
    end

    subgraph Storage ["存储与策略 (Storage & Policies)"]
        G["因子库 (FactorLibrary)"]
        H["内存策略 (MemoryPolicy)"]
        I["提示词构建 (PromptContextBuilder)"]
    end

    subgraph Output ["输出层 (Outputs & Benchmarks)"]
        J["运行时分析 (Runtime Analysis)"]
        K["运行时基准测试 (Runtime Benchmarks)"]
    end

    %% 关系连接
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G

    %% 内存与反馈回路
    D --> H
    H --> I
    I --> D

    %% 输出依赖
    G --> J
    G --> K
    H --> K
    B --> K
```

## 2. 阶段执行流水线 (Stage Pipeline)

展示了单个因子挖掘迭代 (Iteration) 内五个核心阶段的顺序流转以及它们与外围服务的交互。

```mermaid
flowchart LR
    subgraph Pipeline ["迭代流水线 (Iteration Pipeline)"]
        direction LR
        S1["检索阶段\n(RetrieveStage)"] --> S2["生成阶段\n(GenerateStage)"]
        S2 --> S3["评估阶段\n(EvaluateStage)"]
        S3 --> S4["库更新阶段\n(LibraryUpdateStage)"]
        S4 --> S5["提炼阶段\n(DistillStage)"]
    end

    %% 反馈与循环
    S5 -. "开启下一轮" .-> S1

    %% 依赖的服务
    M["内存策略\n(MemoryPolicy)"]
    P["提示词构建\n(PromptContextBuilder)"]
    EK["评估内核\n(EvaluationKernel)"]
    FA["因子准入\n(FactorAdmissionService)"]

    S1 -. "读取经验先验" .-> M
    S2 -. "提供生成上下文" .-> P
    S3 -. "评分与冗余校验" .-> EK
    S4 -. "因子替换/新增" .-> FA
```

## 3. 算法层内部关系图

展示了具体算法（评价内核、几何学、准入机制）之间的复杂协同工作模式，包括如何通过指标（IC/Distance/Spearman）对因子进行过滤与筛选。

```mermaid
flowchart TD
    subgraph Candidate_Processing ["候选因子处理"]
        C1["候选因子表达式 (DSL)"] --> C2["树解析执行 (Tree Parsing)"]
        C2 --> C3["计算预测能力 (Predictive Scoring: IC, 收益)"]
    end

    subgraph Library_Geometry ["库几何与冗余计算 (Library Geometry)"]
        G1["相关性指标\n(Dependence Metric:\nSpearman/Pearson/Distance)"]
        G2["饱和度检测\n(Saturation Diagnosis)"]
        G3["冗余度矩阵\n(Redundancy Matrix)"]
    end

    subgraph Admission_Logic ["准入与替换逻辑 (Admission Logic)"]
        A1["评估内核\n(EvaluationKernel)"]
        A2["因子准入服务\n(FactorAdmissionService)"]
        A3["提炼阶段更新记录\n(DistillStage / LifecycleStore)"]
    end

    %% 候选因子流入评估内核
    C3 --> A1

    %% 评估内核与几何模块交互
    A1 <-->|请求检查相关性| G1
    A1 <-->|检查家族饱和度| G2
    A1 --> G3

    %% 准入服务执行决策
    A1 -->|通过阈值的候选| A2
    G3 -->|如果重复，寻找替换目标| A2

    %% 更新到生命周期
    A2 -->|准入/被拒绝/被替换| A3
```

## 4. 经验内存进化图 (Memory Evolution)

描述 MemoryPolicy 如何将各个阶段的经验提取、演化并输入给下一轮。

```mermaid
flowchart TD
    E1["新准入的因子\n(Successes)"] --> M1["记录成功模式"]
    E2["被拒的因子\n(Failures)"] --> M2["记录失败/高复杂模式"]
    E3["家族发现\n(FactorFamilyDiscovery)"] --> M3["结构化家族分布缺口"]

    M1 --> M4["经验内存\n(Experience Memory)"]
    M2 --> M4
    M3 --> M4

    M4 --> R1["生成检索信号\n(Retrieval Signal)"]
    R1 --> P1["PromptContextBuilder\n(构建 LLM Prompt)"]
```
