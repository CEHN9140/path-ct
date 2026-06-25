# Logic-V-Sub V2 Scaffold

当前工程已进一步按 `logic_v_sub_agent_build_guide_v2.md` 调整为“单入口 Python coordinator + 工具节点”的结构：

- `main.py` 内联 patient / cluster / coordinator 的 LangGraph 定义和主流程
- `tools/` 只保留各个工具节点及其必要后端
- `utils/` 收纳状态 schema、存储、聚类流程和通用辅助代码

## 当前结构

```text
.
├── configs/
├── data/
├── main.py
├── output/
├── tools/
    ├── ct_qc.py
    ├── ct_radiomics.py
    ├── ct_tumor_seg.py
    ├── rna.py
    ├── wsi_embeddings.py
    ├── wsi_patch.py
    ├── wsi_qc.py
    ├── wxs.py
    ├── pathology_qc/
    ├── nnUNet/
    └── prov-gigapath/
└── utils/
    ├── cluster_flow.py
    ├── patient_flow.py
    └── ...
```

## 设计原则

- 真实可执行工具节点保留在 `tools/`。
- 主流程是一个 main graph，固定工作流提出候选亚型簇后进入 subtype review agent。
- QC 与模态流水线保持确定性，不让 LLM 主观决定质控结果。
- `patient_flow / cluster_flow` 等非工具执行逻辑已收口到 `utils/`。
- `router / text / verifier_explainer / report` 的模型位点已预留，但默认不启用 LLM。
- `tools/pathology_qc/`、`tools/nnUNet/`、`tools/prov-gigapath/` 是工具后端依赖，继续保留在 `tools/`。

## 统一项目 State

`main_graph` 和 `subtype_review_graph` 统一使用同一个 `ProjectState`。跨节点和跨阶段流转只保留这 8 个顶层字段：

```text
case_id
inventory
qc
ct_evidence
wsi_evidence
text_evidence
omics_evidence
candidate_cluster_ids
```

其中 `inventory` 只记录输入数据里的原始病例信息，不写入 `has_clinical`、`wsi_records`、`selected_ct` 这类派生清单；`qc` 只取 `success` / `fail`；`ct_evidence`、`wsi_evidence` 只保留数值向量 `features` 和 `feature_path`；`text_evidence` 暂时为空；`omics_evidence` 只保留 `rna_features`、`rna_feature_path`、`wxs_features`、`wxs_feature_path`，RNA/WXS 特征也都是数值向量。各 feature path 仅作为 state 内 features 读取异常时的备选。

## 当前执行顺序

1. `main_graph` 执行 inventory、quality gate、evidence builder、candidate proposer
2. `subtype_review_agent` 对每个候选亚型簇调用 `subtype_review_graph`
3. 汇总 `cluster_reports` 和最终输出

## 运行

```bash
python main.py \
  --data-json-path /data/qijun/path-ct/data/data_test.json \
  --config-dir /data/qijun/path-ct/configs
```

## 配置

工具配置文件放在 `configs/`，入口通过 `--config-dir` 指定配置目录。配置里的相对路径会按 `config_dir` 的上一级目录解析。

- 例如 `--config-dir /data/qijun/path-ct/configs`
- `tools/prov-gigapath` 会解析为 `/data/qijun/path-ct/tools/prov-gigapath`
- `snf.yaml` 控制 SNF 融合的邻居数、迭代次数、`mu`、`alpha`，以及 CT 特征过滤阈值；各模态距离度量由代码固定为 CT 欧氏、WSI 余弦、RNA Spearman、WXS Jaccard。
- `candidate_clustering.yaml` 控制候选簇生成的重复次数、最大聚类数，以及 spectral、hierarchical、PAM 参数。

## 输出

- 如果环境安装了 `langgraph`，`main_graph` 和 `subtype_review_graph` 会用 `draw_mermaid_png()` 保存 PNG 到 `output/graphs/`
- 单病例状态会写到 `output/storage/patient_states/`
- 初始候选亚型簇会写到 `output/candidate_subtype/`
- 各算法候选划分会按算法写到 `output/candidate_subtype/hierarchical/`、`spectral/`、`pam/`
- 共识聚类矩阵、最佳 K、CDF 图、delta area 图和一致性热图会写到 `output/candidate_subtype/consensus_cluster/`
- 簇级复核状态会写到 `output/storage/cluster_states/`
- 最终输出会写到 `output/storage/reports/final_output.json`
- WSI / CT 的真实 QC 结果仍沿用原有 `output/wsi_qc/` 与 `output/ct_qc/`
