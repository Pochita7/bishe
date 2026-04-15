# MAS 与 Guardian Mermaid 流程图

说明：

- 当前 benchmark 主循环使用的是 [mas/team.py](mas/team.py) 的 MASTeam 手工编排。
- 当前 defended benchmark 实际接入的是 Gate 0 预扫描 + Gate 3 ToolGate，入口见 [run_benchmark.py](run_benchmark.py) 与 [run_full_benchmark.py](run_full_benchmark.py)。
- Guardian 的五个 Gate 完整定义在 [tamas_adapter/guardian.py](tamas_adapter/guardian.py)。
- Gate 1、2、4 的端到端串接示例在 [tamas_adapter/agents.py](tamas_adapter/agents.py) 的 SOP 工作流中。

## 1. 当前 MAS 编排循环

```mermaid
flowchart LR
    A[任务<br/>Defended: 先做 Gate 0] --> P[Planner<br/>首轮只出 PLAN]
    P --> X[解析 PLAN<br/>失败则 Selector 兜底]
    X --> W[Worker 组执行<br/>并行 + ToolGate + HANDOFF]
    W --> R{Planner Review<br/>有 FINAL_ANSWER?}
    R -- 是 --> Z[返回答案与指标]
    R -- 否 --> MR{达到 max_rounds?}
    MR -- 否 --> X
    MR -- 是 --> FP[强制 FINAL_ANSWER]
    FP --> Z
```

## 2. Gate 0: ToolDescGate

```mermaid
flowchart LR
    A[工具 name + description] --> B[规则引擎扫描<br/>7 类注入模式]
    B --> C{命中数}
    C -- 0 --> S[SAFE]
    C -- 1 --> U[SUSPECT]
    C -- 2+ --> M[MALICIOUS]

    U --> L{有 llm_client?}
    L -- 否 --> C1[保留 SUSPECT]
    L -- 是 --> R[LLM 语义复核]
    R --> S2[SAFE]
    R --> U2[SUSPECT]
    R --> M

    S --> K[缓存扫描结果]
    S2 --> K
    C1 --> K
    U2 --> K
    M --> D[加入 desc_blocked_tools]
    D --> K
```

## 3. Gate 1: InputGate

```mermaid
flowchart LR
    A[用户输入] --> B[检测 R1 身份冒充<br/>R2 Prompt 注入<br/>R3 恶意意图<br/>R4 高风险指令]
    B --> C{命中规则?}
    C -- 否 --> AL[ALLOW]
    C -- 是 --> D{包含 R2?}

    D -- 是 --> E{strict_mode?}
    E -- 是 --> BL[BLOCK]
    E -- 否 --> SI[sanitize_injection]
    SI --> SA[SANITIZE]

    D -- 否 --> F{包含 R1?}
    F -- 是 --> RI[remove_impersonation]
    RI --> SA

    F -- 否 --> G{包含 R3 或 R4?}
    G -- 否 --> AL
    G -- 是 --> SM[sanitize_malicious_directives]
    SM --> H{文本有变化?}
    H -- 是 --> SA
    H -- 否 --> AL2[ALLOW<br/>仅记录风险]
```

## 4. Gate 2: PlanGate

```mermaid
flowchart LR
    A[Planner 输出计划] --> B[检查 P1<br/>是否引用黑名单工具]
    B --> C[检查 P2<br/>是否出现可疑操作序列]
    C --> D[检查 P3<br/>是否出现 Byzantine 工具]
    D --> E{命中类型}

    E -- 无 --> AL[ALLOW]
    E -- P1 --> S1[替换恶意工具为<br/>BLOCKED:tool]
    S1 --> SA1[SANITIZE]
    E -- 仅 P3 --> SA2[SANITIZE]
    E -- 仅 P2 --> AL2[ALLOW<br/>可疑但未确认]
```

## 5. Gate 3: ToolGate

```mermaid
flowchart LR
    A[工具调用 name,args] --> B{在黑名单?}
    B -- 是 --> BL1[BLOCK]
    B -- 否 --> C{被 ToolDescGate<br/>标记恶意?}
    C -- 是 --> BL2[BLOCK]
    C -- 否 --> D{敏感工具?}

    D -- 否 --> AL[ALLOW]
    D -- 是 --> E[检查参数关键词]
    E --> F{参数异常?}
    F -- 是 --> BL3[BLOCK]
    F -- 否 --> AL

    BL1 --> R1[wrapper 返回<br/>access denied]
    BL2 --> R1
    BL3 --> R1
    AL --> X[执行真实工具]
```

## 6. Gate 4: OutputGate

```mermaid
flowchart LR
    A[Executor 输出] --> B[检查 O1 恶意结果特征<br/>检查 O2 越权操作声明]
    B --> C{命中规则?}
    C -- 否 --> AL[ALLOW<br/>继续到 Verifier]
    C -- 是 --> BL[BLOCK]
    BL --> S[替换为安全 RESULT<br/>或直接阻断下游]
    S --> V[Verifier]
    AL --> V
```

## 7. 五个 Gate 串接总览

```mermaid
flowchart LR
    Q[User Query] --> G1[Gate 1<br/>InputGate]
    G1 --> P[Planner]
    P --> G2[Gate 2<br/>PlanGate]
    G2 --> E[Executor]
    E --> G3[Gate 3<br/>ToolGate]
    G3 --> E
    E --> G4[Gate 4<br/>OutputGate]
    G4 --> V[Verifier]
    T[工具描述预扫描] --> G0[Gate 0<br/>ToolDescGate]
    G0 --> G3
```