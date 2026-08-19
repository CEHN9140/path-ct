# ccRCC候选亚型Review Agent重构规范
## 基于主流多组学/多模态肿瘤亚型研究的验证指标与动作决策规则

**版本：v2.0（Codex实施版）**  
**日期：2026-08-19**  
**项目：基于多模态数据的肾透明细胞癌ccRCC新亚型发现**  
**数据：TCGA-KIRC；CT、病理WSI、RNA-seq、WXS、CNV、Clinical**  
**当前发现阶段：四模态patient-similarity network（CT、WSI、RNA、Genomic=WXS+CNV）→SNF→候选partition→Review Agent**

---

# 0.本文档的目的与非目标

本文档用于**直接交给Codex重构当前`subtype_review`模块**。目标不是重新设计上游数据处理，也不是让LLM重新做聚类，而是重构：

1. 科学验证维度；
2. 每个维度的指标；
3. Split/Merge proposal的生成与验证方式；
4. `accept/drop/split/merge/need_more_evidence`动作规则；
5. Evidence schema、缓存/失效规则和终态语义。

## 0.1 本轮不得修改的上游流程

除非发现明确代码bug，否则不要修改：

- CT QC、phase判定、CT selection；
- CT肿瘤分割；
- WSI QC、patch、PHARAOH肿瘤ROI；
- CT radiomics特征工程；
- Prov-GigaPath WSI embedding；
- RNA、WXS、CNV上游特征工程；
- WXS+CNV内部Genomic fusion；
- 四模态SNF；
- Candidate K选择及当前candidate partition输入。

本轮只重构`subtype_review`科学协议、tools、evidence registry和动作逻辑。

---

# 1.最终设计结论

## 1.1 Validation dimensions从5个改为4个

删除：

```text
structural_adequacy
```

作为独立validation dimension。

保留：

```text
biological_support
cross_modal_consistency
confounder_exclusion
known_label_echo
```

这4个维度分别回答：

| Dimension | 科学问题 |
|---|---|
| `biological_support` | 每个集合是否具有可解释的生物学特征；集合/子集合之间是否存在有意义的biology difference？ |
| `cross_modal_consistency` | 同一批患者是否在CT、WSI、RNA、Genomic各自的patient representation中得到一致或互补支持？ |
| `confounder_exclusion` | 当前集合或集合边界是否主要由center、scanner、protocol、phase、batch等technical factors解释？ |
| `known_label_echo` | 当前taxonomy或结构修正是否只是重新发现stage、grade或已有ccRCC molecular taxonomy？ |

---

# 2.Structural adequacy不再是validation dimension

## 2.1 保留“结构候选生成”能力

当前`tool_structural_adequacy.py`里有价值的部分不要删除，而应重构为：

```text
Structure Proposal Generator
```

建议文件：

```text
tools/structure_proposal_generator.py
```

它是**确定性Python模块**，不是Verifier capability，不生成`supporting/conflicting`scientific finding。

它只负责：

### Split proposals
例如：

```text
split:C0003:k2
C0003 -> A(12) + B(18)
```

### Merge proposals
例如：

```text
merge:C0001+C0002
```

并给出：

- exact memberships；
- `proposal_id`；
- `source_signature`；
- basic generator metrics；
- 是否`eligible_for_review`。

---

# 3.为什么Structure Proposal Generator必须在Router动作之前运行

旧逻辑：

```text
Verifier
→ Router先决定Split/Merge
→ Plan Generator
→ Reviser
```

新逻辑必须改成：

```text
Partition
→ Structure Proposal Generator
→ Verifier针对具体proposal获取四维证据
→ Router决定Split/Merge/Accept/Drop/NeedMoreEvidence
→ Reviser从已经验证过的legal proposals中选plan
→ Executor
```

原因：

Router要判断`Split`，必须先知道：

- proposed child membership究竟是什么；
- 同一个A/B划分在各原始模态是否存在；
- A/B是否有biology差异；
- A/B是否只是scanner/phase；
- A/B是否只是已有known label。

因此proposal必须先存在。

---

# 4.增加Evidence Scope：4维×3个主要scope

每一个validation dimension不能再只输出一个模糊的`supporting/conflicting`。

至少支持：

```python
EvidenceScope = Literal[
    "set_identity",
    "split_proposal",
    "merge_proposal",
    "partition",
]
```

其中：

- `set_identity`：服务Accept/Drop；
- `split_proposal`：服务Split；
- `merge_proposal`：服务Merge；
- `partition`：主要用于whole-partition known-label comparison、whole-partition modality PERMANOVA等。

---

# 5.新的Evidence schema

建议：

```python
class AuditFinding(BaseModel):
    dimension: Literal[
        "biological_support",
        "cross_modal_consistency",
        "confounder_exclusion",
        "known_label_echo",
    ]
    scope: Literal[
        "set_identity",
        "split_proposal",
        "merge_proposal",
        "partition",
    ]
    target_ids: list[str]
    proposal_id: str | None = None
    subject_signature: str

    status: Literal[
        "supporting",
        "conflicting",
        "mixed",
        "inconclusive",
        "unavailable",
    ]

    summary: str
    metric_refs: list[str]
```

`EvidenceGap`同样增加：

```python
dimension
scope
target_ids
proposal_id
subject_signature
reason
```

---

# 6.为什么必须引入`scope`

例如：

```text
cross_modal_consistency = supporting
```

至少可能表示：

1. `set_identity`：C0003作为整体在多个模态里有共同identity；
2. `split_proposal`：C0003的12/18划分被多个模态共同看到；
3. `merge_proposal`：C0001/C0002原边界在多个模态中很弱，支持合并。

这三种语义完全不同，不能继续混用。

---

# 7.Structure Proposal Generator：Split

## 7.1 Proposal生成

对active set `Si`取fused SNF affinity子矩阵，生成少量局部候选：

```text
k_local = 2
k_local = 3
...
```

第一版继续沿用当前：

```text
SpectralClustering(affinity="precomputed")
```

即可。

约束：

```yaml
min_split_size: 10
max_split_children: 4
max_split_proposals_per_set: 3
```

---

## 7.2 Proposal Generator可保留的结构统计

保留当前有价值的：

- child sizes；
- normalized separation；
- minimum child separation；
- selection-adjusted edge-permutation null；
- observed separation；
- null mean/CI；
- gain over null；
- permutation p；
- BH-q。

其科学用途仅为：

> “这个局部substructure是否足够值得进入四维科学审查？”

不是：

> “已经证明应该Split。”

---

## 7.3 Split proposal的`eligible_for_review`

建议：

```python
eligible_for_review = (
    min(child_sizes) >= min_split_size
    and fused_selection_adjusted_q <= 0.05
    and fused_gain_over_null > 0
)
```

注意：

```text
eligible_for_review != Split
```

---

## 7.4 permutation次数

开发/debug：

```yaml
split_null_permutations: 199
```

最终正式运行：

```yaml
split_null_permutations: 1999
```

计算允许可升至4999。

199次时最小经验p为：

```text
1 / (199 + 1) = 0.005
```

适合开发，不适合最终精细推断。

---

# 8.Structure Proposal Generator：Merge

当前K通常较小，第一版直接为所有active set pair生成proposal：

```text
merge:Si+Sj
```

不要使用：

```text
p > 0.05
```

作为生成Merge proposal的条件。

第一版只生成pairwise merge。

若以后需要3个集合合并，通过迭代pairwise merge完成，便于审计和证据失效管理。

---

# 9.四维×三Scope总矩阵

| Dimension | `set_identity` | `split_proposal` | `merge_proposal` |
|---|---|---|---|
| Biological | 这个集合有什么biology identity？ | parent内部children是否biology不同？ | 两个集合是否存在强biology distinction？ |
| Cross-modal | 这个集合是否在多个模态中被共同表征？ | 同一个child划分能否跨模态被看到？ | 两集合原边界在多个模态中是否仍有必要？ |
| Confounder | 集合是否主要由technical factor驱动？ | proposed child boundary是否由technical factor解释？ | 原Si/Sj边界是否由technical factor解释？ |
| Known-label | 当前taxonomy是否重复已有标签？ | Split后是否近似恢复known taxonomy？ | Merge后是否改变/形成known taxonomy echo？ |

---

# 10.Biological support：总体职责

`biological_support`回答：

> 每个候选集合是否具有可解释的分子生物学identity；如果考虑Split，children之间是否存在biology differentiation；如果考虑Merge，两个集合之间是否存在足以阻止合并的强biological distinction。

至少包含：

```text
RNA pathway
WXS mutation
CNV alteration / burden
```

当前只有RNA+WXS不够，本轮必须把CNV正式加入biology。

---

# 11.Biology / Set Identity

## 11.1 RNA pathway

建议继续使用Hallmark sample-level pathway scores：

```text
GSVA或ssGSEA
```

每个set：

```text
Si vs Rest
```

输出：

- set mean/median；
- rest mean/median；
- delta；
- Mann-Whitney U；
- Cliff's delta；
- SMD作为secondary；
- BH-q；
- direction。

必须输出top biologically interpretable pathways，而不仅仅是：

```text
significant_count_q05
```

---

## 11.2 RNA gene-level

可保留DEG作为解释层，但不是主要identity指标。

如果使用当前规范化连续表达：

- effect；
- Mann-Whitney或适合的模型；
- BH-q。

如果以后改用raw counts做正式DEG，应使用专门RNA-seq differential expression framework。

本轮不要求重做上游RNA。

---

## 11.3 WXS

每个gene/feature：

```text
Si vs Rest
```

输出：

- mutation prevalence；
- rest prevalence；
- Δfrequency；
- Fisher exact p；
- odds ratio；
- OR 95%CI；
- BH-q。

必须保留方向。

例如OR<1表示depletion，不得写成enrichment。

---

## 11.4 CNV

正式加入biology。

### arm/locus alteration

输出：

- gain/loss prevalence；
- Δfrequency；
- Fisher；
- OR；
- OR CI；
- BH-q。

### global burden

例如：

- FGA；
- gain burden；
- loss burden。

输出：

- median/mean；
- delta；
- Mann-Whitney；
- Cliff's delta；
- BH-q。

---

# 12.Biology / Split Proposal

若proposal为：

```text
parent Si -> child A + child B
```

真正检验必须是：

```text
A vs B within parent
```

不能再使用：

```text
A vs entire cohort rest
B vs entire cohort rest
```

来证明Split。

---

## 12.1 RNA

- A vs B pathway scores；
- effect；
- Cliff's delta；
- p/q。

## 12.2 WXS

- A vs B mutation prevalence；
- Fisher；
- OR；
- Δfrequency；
- q。

## 12.3 CNV

- A vs B arm/locus alteration；
- Fisher/OR/q；
- burden difference；
- Cliff/q。

---

## 12.4 Biology Split finding语义

### `supporting`

至少存在一组：

```text
FDR controlled
+
material effect
+
biologically coherent
```

的child differences。

### `inconclusive`

当前RNA/WXS/CNV没有得到足够清晰的children differentiation。

**禁止：**

```text
inconclusive -> Split false
```

### `conflicting`

第一版慎用。

除非未来实现严格practical-equivalence分析，否则“不显著”不能作为biology conflicting。

---

# 13.Biology / Merge Proposal

核心原则：

```text
p > 0.05 != similarity
p > 0.05 != equivalence
```

因此Biology在Merge第一版主要担当：

```text
distinctness veto
```

即：

> 若Si/Sj存在强、稳定、可解释的biology difference，则阻止Merge。

---

## 13.1 RNA profile

构建每个set的Hallmark centroid：

```text
B_i = [mean(pathway1), ..., mean(pathwayP)]
```

输出：

- Spearman correlation；
- cosine similarity；
- pathway-wise effect/q；
- top distinct pathways。

---

## 13.2 WXS profile

mutation prevalence vector：

输出：

- profile similarity；
- gene-wise OR/q；
- top distinct mutations。

---

## 13.3 CNV profile

输出：

- arm/locus prevalence profile correlation；
- burden difference；
- top distinct CNVs。

---

## 13.4 Merge Biology finding

### `conflicting`

存在strong coherent biological distinction。

### `inconclusive`

未发现明确difference，但也没有严格equivalence证据。

### `supporting`

第一版尽量不使用。

即：

> Biology主要阻止不合理Merge，不用“不显著”积极支持Merge。

---

# 14.Cross-modal consistency：总体职责

回答：

> 同一个patient set或同一个structure proposal，在CT、WSI、RNA、Genomic各自独立patient representation中能否得到一致或互补支持？

只使用单模态patient affinity/distance表示：

```text
CT
WSI
RNA
Genomic
```

不要为这一维度重新concat原始features。

---

# 15.Cross-modal / Set Identity

每个modality分别计算。

---

## 15.1 Per-patient / per-set Silhouette

使用当前partition labels，在该modality distance space计算：

```text
silhouette_i
```

每set输出：

- mean silhouette；
- median silhouette；
- p10/p25；
- fraction(silhouette > 0)。

Silhouette是标准的tightness/separation指标。

---

## 15.2 Own-set affinity margin

对patient `i`：

```text
own_affinity =
mean affinity(i, members of own set excluding self)

best_other_affinity =
max over other sets mean affinity(i, other set)

margin =
own_affinity - best_other_affinity
```

每set输出：

- mean/median margin；
- p10/p25；
- fraction(margin > 0)。

这是本项目network-specific effect metric。

必须在文档/论文中明确定义，不得冒充已有经典统计量。

---

## 15.3 Within / nearest-between separation

保留当前：

- mean within affinity；
- nearest other set；
- mean nearest-between affinity；
- normalized separation。

---

## 15.4 Partition-level PERMANOVA

当前代码只有distance-based R²，没有真正permutation p。

重构为真正PERMANOVA：

- pseudo-F；
- R²；
- permutation p；
- BH-q（4 modalities作为family）。

建议：

```yaml
permanova_permutations_dev: 199
permanova_permutations_final: 1999
```

若不实现p-value，则必须把字段改名为：

```text
distance_based_r2
```

不得继续把不完整计算称为完整PERMANOVA。

---

# 16.Cross-modal / Split Proposal

给定固定：

```text
A/B membership
```

必须在CT、WSI、RNA、Genomic中检查**同一个A/B**。

不能：

```text
CT自己找A/B
WSI自己找C/D
```

然后因为都能K=2就说“一致”。

---

## 16.1 每个modality输出

- child mean/median silhouette；
- normalized A/B separation；
- minimum child separation；
- fixed-label permutation p；
- q；
- optional local-K2 ARI with proposal；
- optional local-K2 AMI with proposal。

---

## 16.2 单modality Split support flag

推荐：

```python
modality_split_support = (
    q_value <= 0.05
    and normalized_separation > 0
    and median_silhouette > 0
)
```

不要设：

```text
silhouette > 0.5
```

这类没有本项目依据的高阈值。

---

## 16.3 Cross-modal Split支持规则

建议项目default：

```yaml
split_min_supporting_modalities: 2
split_require_molecular_or_biology: true
```

即：

### 条件A
至少2个原始模态支持**同一个proposal membership**。

### 条件B
至少满足一个：

```text
RNA supports split
OR
Genomic supports split
OR
Biology/split_proposal = supporting
```

这样：

```text
CT + WSI only
```

的Split不会直接升级成“新的ccRCC肿瘤亚型”。

它应先请求biology/molecular evidence。

这是项目规则，不是领域统一定理。

---

# 17.Cross-modal / Merge Proposal

对：

```text
Si, Sj
U = Si ∪ Sj
```

只比较当前Si/Sj边界。

每modal输出：

- pairwise silhouette；
- own-set affinity margin；
- within affinity；
- between affinity；
- normalized boundary separation。

---

## 17.1 单modality Merge support

保守default：

```python
modality_merge_support = (
    median_pairwise_silhouette <= 0
    and median_own_set_affinity_margin <= 0
)
```

语义：

> 在这个modality里，患者平均意义上没有明显更靠近自己的原集合。

如果：

```text
silhouette = +0.03
margin = +0.01
```

第一版标记`inconclusive`，不要积极Merge。

---

## 17.2 Cross-modal Merge支持规则

建议：

```yaml
merge_min_supporting_modalities: 2
```

要求：

1. 至少2个原始模态明确弱边界；
2. 不能有>=2个原始模态明确支持维持原边界；
3. Biology不能存在strong distinction veto；
4. Confounder可加强Merge解释但不能单独决定。

核心：

```text
positive weak-boundary evidence -> Merge support
```

而不是：

```text
no significant difference -> Merge support
```

---

# 18.Confounder exclusion：总体职责

回答：

> 当前set identity或结构proposal是否主要由非生物technical factors解释？

重点：

```text
center
scanner
phase
protocol
reconstruction
platform
batch
processing
```

而不是把所有临床变量都叫technical confounder。

---

# 19.Confounder字段

## CT至少包括

- phase；
- phase_group；
- manufacturer；
- scanner_model；
- reconstruction_kernel；
- slice_thickness；
- pixel_spacing；
- z_spacing；
- n_images；
- study_year。

## WSI如metadata可靠可包括

- slide scanner；
- scanning site；
- magnification；
- staining/site batch；
- preparation metadata。

## RNA/WXS/CNV如GDC metadata可靠可包括

- sequencing center；
- platform；
- plate/batch；
- pipeline/version；
- center/site。

无法可靠获得则标`unavailable`，不要猜。

---

# 20.Confounder统计指标

## 20.1 categorical

主输出：

- contingency table；
- chi-square（expected count足够）；
- sparse table使用exact/Monte-Carlo/permutation；
- **bias-corrected Cramér's V**；
- BH-q。

普通Cramér's V在小样本+高维contingency table中存在上偏，应使用bias-corrected版本。

---

## 20.2 numeric

主输出：

- Kruskal-Wallis；
- BH-q；
- epsilon-squared；
- pairwise Cliff's delta。

当前SMD和raw eta²可以保留为secondary descriptive metrics。

---

# 21.Confounder / Set Identity

检查：

```text
current set / current partition labels
vs
technical factors
```

只有：

```text
significance
+
large/material effect
```

才能形成强`conflicting`。

单纯：

```text
p<0.05
```

不够。

---

# 22.Confounder / Split Proposal

只在parent内部：

```text
A/B child labels
vs
technical factors
```

例如：

```text
A几乎全部NEPH
B几乎全部DEL
```

或：

```text
A几乎全部GE
B几乎全部Siemens
```

则Split具有强technical explanation。

这可以阻止Split。

---

# 23.Confounder / Merge Proposal

检查原：

```text
Si/Sj boundary
```

是否高度对应technical factor。

如果：

```text
Si ≈ GE
Sj ≈ Siemens
```

而Cross-modal本身又显示边界弱，则technical explanation可增强Merge。

但：

```text
technical association alone != Merge
```

---

# 24.Confounder project-default effect thresholds

为了Codex deterministic flag可执行，可配置：

```yaml
confounder:
  alpha: 0.05
  categorical_strong_v: 0.50
  numeric_large_cliffs_delta: 0.474
```

注意：

- 这些不是全领域统一标准；
- 是项目预注册式default；
- 最终supplement做threshold sensitivity。

---

# 25.Known-label echo：总体职责

回答：

> 当前taxonomy或proposed revision是否只是重新发现已有标签？

核心区别：

```text
association != echo
```

例如C0003更多Stage IV不等于C0003是Stage IV echo。

Echo要求：

> 整体partition结构接近已有taxonomy的一一映射。

---

# 26.Known-label指标

保留：

- AMI；
- homogeneity；
- completeness；

增加：

- ARI；
- optimal label mapping accuracy；
- contingency table。

---

# 27.Known-label参考体系

当前：

```text
stage
grade
```

必须保留。

若论文声称：

```text
novel ccRCC subtype
```

应尽量增加：

```text
TCGA mRNA m1–m4
ccA/ccB
其他可可靠重建的公认ccRCC molecular taxonomy
```

无法可靠映射的known label不要硬造。

---

# 28.Known-label / Set Identity / Partition

对于当前partition `P`与known taxonomy `L`：

输出：

- AMI；
- ARI；
- homogeneity；
- completeness；
- optimal mapping accuracy。

---

# 29.Known-label / Split Proposal

计算：

```text
P_current vs L
P_split vs L
```

输出：

```text
AMI_current
AMI_split
ΔAMI

ARI_current
ARI_split
ΔARI

mapping accuracy变化
```

如果Split后突然近似一一恢复已知分类，则：

```text
known_label_echo/split_proposal = conflicting
```

---

# 30.Known-label / Merge Proposal

同理：

```text
P_current vs L
P_merge vs L
```

检查：

- ΔAMI；
- ΔARI；
- homogeneity/completeness；
- mapping accuracy。

Known-label通常不主动要求Merge，而用于判断：

> Merge后是否只是回到known taxonomy，或原两个set是否分别对应已有实体。

---

# 31.Known-label near-identity project policy

没有公认：

```text
AMI >= 0.8 = echo
```

这样的万能阈值。

为了Codex可执行，建议项目default：

```yaml
known_label_echo:
  near_identity_ami: 0.80
  near_identity_ari: 0.80
  near_identity_mapping_accuracy: 0.90
```

强echo建议：

```python
near_echo = (
    (AMI >= 0.80 or ARI >= 0.80)
    and optimal_mapping_accuracy >= 0.90
)
```

正式supplement建议敏感性：

```text
0.70 / 0.80 / 0.90
```

必须在论文中写成project rule，而不是“领域标准”。

---

# 32.FDR family统一规则

使用Benjamini-Hochberg。

## Biology identity

分别将：

```text
all active sets × Hallmark pathways
all active sets × WXS features
all active sets × CNV features
```

作为三个BH family。

不要再每个set完全独立BH后，把不同set的q拿来共同做动作决策。

---

## Biology Split

同一proposal：

- pathways作为一个family；
- WXS一个family；
- CNV一个family。

如果同一parent保留多个split proposals并并行评估，可把：

```text
proposal × feature
```

纳入同一parent-level family。

---

## Cross-modal Split

同一parent的所有split proposals，在每个modality内部BH。

---

## Confounder

同一scope下全部technical factors作为一个family。

---

# 33.动作层：总体原则

Router仍只允许：

```text
need_more_evidence
accept
drop
split
merge
```

但新增一个非Router动作的合法terminal：

```text
unresolved
```

---

# 34.动作优先逻辑

推荐：

```text
1. 还有decision-changing evidence没查？
   -> need_more_evidence

2. 有已经充分支持的结构修正？
   -> split / merge

3. 当前set具有足够独立identity？
   -> accept

4. 有积极反证表明它不应作为独立新亚型？
   -> drop

5. 证据穷尽且没有动作成立
   -> terminal unresolved
```

原则：

```text
结构修正优先于身份淘汰
```

---

# 35.Accept规则

Accept回答：

> 这个set是否足以作为独立candidate subtype保留？

建议hard contract：

1. 无未解决actionable Split；
2. 无未解决actionable Merge；
3. `confounder_exclusion/set_identity != conflicting`；
4. `known_label_echo/set_identity != conflicting`；
5. `cross_modal_consistency/set_identity = supporting`；
6. Biology允许：
   - supporting
   - mixed
   - inconclusive

**删除：**

```text
Accept requires biological_support=supporting
```

---

# 36.Cross-modal identity最低project rule

建议：

至少2个原始模态满足：

```text
median silhouette > 0
AND
normalized separation > 0
AND
own-set affinity support fraction > 0.5
```

即：

```yaml
accept_min_supporting_modalities: 2
```

不要求4个模态全通过。

---

# 37.Split动作规则

Router允许Split必须满足：

### 37.1 legal proposal存在

```text
eligible_for_review = true
```

即至少：

```text
min child >= 10
fused selection-adjusted q <= 0.05
fused gain > 0
```

### 37.2 Cross-modal Split positive

至少2个原始模态支持同一child membership：

```text
q <= 0.05
separation > 0
median silhouette > 0
```

### 37.3 molecular/biology entry

至少一个：

```text
RNA supports
OR
Genomic supports
OR
Biology/split_proposal = supporting
```

### 37.4 no technical veto

```text
confounder_exclusion/split_proposal != conflicting
```

### 37.5 no strong known-label echo

```text
known_label_echo/split_proposal != conflicting
```

---

# 38.Imaging-only Split的处理

如果：

```text
CT supports
WSI supports
RNA weak
Genomic weak
Biology split not attempted
```

则不要直接Split。

Router应：

```text
need_more_evidence
dimension=biological_support
scope=split_proposal
proposal_id=...
```

如果biology仍inconclusive，则由Router结合全部证据决定：

- 保留parent；
- 或terminal unresolved；

第一版不要把纯imaging split直接命名为新的ccRCC molecular/multimodal subtype。

---

# 39.Merge动作规则

Merge必须有**积极弱边界证据**。

建议：

1. >=2 modalities：
   ```text
   median pairwise silhouette <= 0
   AND
   median own-set affinity margin <= 0
   ```
2. 不能有>=2 modalities强支持维持原边界；
3. `biological_support/merge_proposal != conflicting`；
4. `known_label_echo`不提示合并会造成明显known taxonomy问题；
5. Confounder如解释原边界，可增强Merge，但不能单独决定。

---

# 40.禁止Merge规则

以下均不允许单独支持Merge：

```text
p > 0.05
q > 0.05
没有显著DEG
没有显著pathway
没有显著mutation
```

“不显著”不是“等效”。

---

# 41.Drop动作规则

Drop必须重新定义为：

> 有积极反证表明该set不应作为独立的“新亚型”保留，并且Split/Merge等更合理结构修正已经排除或耗尽。

---

## 41.1 合法Drop理由

### A.强technical artifact

例如：

```text
set identity几乎由scanner/protocol/batch解释
```

且没有合理结构修复。

### B.强known-label near-echo

例如：

```text
near-identity AMI/ARI
+
high mapping accuracy
```

同时没有额外的独立multimodal/biological identity。

---

## 41.2 禁止Drop理由

以下任何一个都不能单独Drop：

```text
RNA q > 0.05
WXS q > 0.05
CNV q > 0.05
某一个modality weak
tool failure
budget pressure
max rounds
biology inconclusive
```

---

## 41.3 Drop的患者语义

Drop仅表示：

```text
drop_from_novel_subtype_catalog
```

不能真正删除患者。

患者必须继续保留在：

- partition history；
- revision log；
- final audit；
- dropped set registry。

---

# 42.Need_more_evidence规则

只有：

```text
至少两个动作仍合理
AND
一个尚未attempt的证据可以现实地改变选择
```

才允许request。

请求必须指定：

```text
dimension
scope
target_ids
proposal_id
```

示例：

```json
{
  "action": "need_more_evidence",
  "dimension": "confounder_exclusion",
  "scope": "split_proposal",
  "target_ids": ["C0003"],
  "proposal_id": "split:C0003:k2",
  "reason": "The proposed split is supported in multiple modalities, but a technical explanation has not yet been excluded."
}
```

---

# 43.Unresolved终态

当：

```text
所有decision-relevant evidence已attempted
AND
无valid Split
AND
无valid Merge
AND
Accept证据不足
AND
也没有positive Drop evidence
```

则：

```text
terminal status = unresolved
```

不能强迫：

```text
Drop
```

---

# 44.新的Finding解释规则

## Biological

### identity
- supporting：有明确biology identity；
- mixed：有部分特征但不一致；
- inconclusive：没有清晰annotation；
- conflicting：慎用。

### split
- supporting：children有清晰biology differentiation；
- inconclusive：未找到清晰差异；
- conflicting：只有严格等效性/强同质性证据才用。

### merge
- conflicting：存在强biological distinction；
- inconclusive：不能证明相似，也没有强区别；
- supporting：第一版慎用。

---

# 45.Cross-modal Finding

## identity

### supporting
多个模态均表现为：

```text
positive silhouette
positive affinity margin
positive separation
```

### mixed
部分模态支持、部分弱。

### conflicting
多个原始模态明显不支持当前set identity。

---

## split

### supporting
同一个child membership被多个原始模态支持。

### conflicting
多个原始模态明确不支持proposal、保持parent整体或形成不同结构。

---

## merge

### supporting
多个模态显示原Si/Sj边界很弱。

### conflicting
多个模态具有清晰positive boundary，支持继续分开。

---

# 46.Confounder Finding

### supporting
没有足够技术因素解释当前identity/proposal。

### conflicting
技术因素同时满足：

```text
FDR significance
+
large/material effect
```

并高度解释当前set/boundary。

### mixed
有关联但效应不足以解释全部结构。

---

# 47.Known-label Finding

### supporting
明显cross-cutting，不是已有taxonomy的一一映射。

### mixed
有关联，但不是简单重复。

### conflicting
near-identity echo。

---

# 48.Evidence生命周期：必须修复whole-partition signature问题

当前逻辑：

```text
任何一个set改变
→ whole partition signature改变
→ 所有旧evidence失效
```

必须修改。

---

# 49.新的subject-local signature

## set identity

```python
set_signature = hash(sorted(member_ids))
```

如果C0001Split，但C0003成员没变：

```text
C0003 identity evidence继续有效
```

---

## split proposal

```python
split_signature = hash(
    parent_members
    + exact_child_groups
)
```

---

## merge proposal

```python
merge_signature = hash(
    exact memberships of sets to be merged
)
```

---

## partition scope

whole-partition known-label等仍使用：

```text
whole partition signature
```

因为taxonomy改变确实会改变AMI/ARI。

---

# 50.Attempted evidence key

不能再只用：

```text
dimension + partition_signature
```

必须使用：

```text
dimension
+ scope
+ subject_signature
+ proposal_id
```

例如：

```text
biological_support
|split_proposal
|hash(...)
|split:C0003:k2
```

---

# 51.当前代码文件重构清单

## `agents/subtype_review/schemas.py`

删除：

```python
"structural_adequacy"
```

增加：

```text
EVIDENCE_SCOPES
scope
proposal_id
subject_signature
```

Router的`need_more_evidence`增加scope/proposal。

---

# 52.`agents/subtype_review/tools.py`

删除：

```text
structural_adequacy validation capability
```

保留：

```text
biological_support
cross_modal_consistency
confounder_exclusion
known_label_echo
```

`execute_capability`需要接收：

```text
scope
target_ids
proposal_id
proposal payload
```

---

# 53.`tools/tool_structural_adequacy.py`

重构/拆分成：

```text
tools/structure_proposal_generator.py
```

只负责proposal。

不要再返回scientific support/conflict。

---

# 54.`tools/tool_multimodal_consistency_check.py`

重构支持：

```text
set_identity
split_proposal
merge_proposal
partition
```

新增：

### Identity
- silhouette；
- affinity margin；
- support fraction；
- true PERMANOVA。

### Split
- fixed child membership silhouette；
- separation；
- fixed-label permutation；
- optional ARI/AMI。

### Merge
- pairwise silhouette；
- affinity margin；
- boundary separation。

---

# 55.`tools/tool_pathway_enrichment.py`

支持：

```text
identity: set vs rest
split: child vs child within parent
merge: pairwise profile + distinctness
```

增加Cliff's delta。

---

# 56.`tools/tool_mutation_enrichment.py`

支持三scope。

增加：

- OR CI；
- explicit direction；
- merge profile/distinctness。

---

# 57.新增CNV biology tool

建议：

```text
tools/tool_cnv_characterization.py
```

或者统一：

```text
tools/tool_genomic_characterization.py
```

第一版倾向独立tool，更便于审计：

```text
biological_support
├── pathway
├── mutation
└── cnv
```

---

# 58.`tools/tool_confound_test.py`

修改：

1. bias-corrected Cramér's V；
2. sparse exact/Monte Carlo/permutation；
3. numeric epsilon²；
4. pairwise Cliff's delta；
5. 支持三scope；
6. 扩展WSI/omics technical metadata（仅可靠字段）。

---

# 59.`tools/tool_known_label_echo_test.py`

增加：

- ARI；
- optimal mapping accuracy；
- proposed partition comparison；
- ΔAMI；
- ΔARI。

---

# 60.`agents/subtype_review/prompts/protocol.md`

删除旧规则：

```text
Accept requires supporting biological_support
```

删除：

```text
biological inconclusive -> Drop
```

删除：

```text
Split/Merge requires structural_adequacy=conflicting
```

改成本文件动作规则。

---

# 61.`agents/subtype_review/prompts/verifier.md`

Verifier必须严格按：

```text
dimension + scope + subject/proposal
```

审计。

禁止：

- generator metrics直接变成validation finding；
- biology无显著结果写成conflicting；
- p>0.05写成Merge support；
- technical p<0.05但effect很小就写成conflicting；
- stage/grade association写成echo；
- 直接推荐动作。

---

# 62.`agents/subtype_review/prompts/router.md`

Router输入：

- active sets；
- active structure proposals；
- identity evidence；
- proposal-specific evidence；
- gaps；
- attempted request keys；
- blocked actions；
- budget。

Router禁止：

- biology inconclusive直接Drop；
- p>0.05直接Merge；
- actionable Split/Merge存在时先Drop parent；
- 自己读取raw matrices算新统计量。

---

# 63.`agents/subtype_review/prompts/reviser.md`

Reviser只能从：

```text
supported_legal_proposals
```

选。

不能重新判断：

```text
should split?
should merge?
```

只选择：

```text
which supported plan?
```

---

# 64.`agents/subtype_review/graph.py`

重点：

1. 移除structural validation capability；
2. 增加proposal generation node；
3. evidence registry改local signature；
4. attempted key改dimension+scope+signature；
5. Accept validator删除biology hard gate；
6. Split/Merge validator要求proposal-specific evidence contract；
7. Drop validator要求positive invalidating evidence；
8. 增加unresolved terminal；
9. status语义改造。

---

# 65.Final status语义

建议：

```text
review_complete_all_accepted
review_complete_with_dropped_sets
review_complete_with_unresolved_sets
review_unavailable
review_failed_runtime
```

禁止：

```text
科学上Drop一个set
→ final_validation_failed
```

科学结论不是程序failure。

---

# 66.Config建议

```yaml
review:
  max_rounds: 20
  max_failures: 3

structure_proposals:
  min_split_size: 10
  max_split_children: 4
  max_split_proposals_per_set: 3
  split_null_permutations_dev: 199
  split_null_permutations_final: 1999

cross_modal:
  permanova_permutations_dev: 199
  permanova_permutations_final: 1999

  accept_min_supporting_modalities: 2

  split_min_supporting_modalities: 2
  split_require_molecular_or_biology: true

  merge_min_supporting_modalities: 2

confounder:
  alpha: 0.05
  categorical_strong_v: 0.50
  numeric_large_cliffs_delta: 0.474

known_label_echo:
  near_identity_ami: 0.80
  near_identity_ari: 0.80
  near_identity_mapping_accuracy: 0.90

statistics:
  alpha: 0.05
  fdr_method: fdr_bh
```

所有非`alpha=0.05`阈值必须在代码注释/config说明中标：

```text
project protocol threshold
```

不能声称是领域统一标准。

---

# 67.统一tool output格式

建议：

```python
{
  "tool_name": "...",
  "status": "success",

  "dimension": "...",
  "scope": "...",
  "target_ids": [...],
  "proposal_id": None or "...",
  "subject_signature": "...",

  "metrics": {...},

  "decision_metrics": {
      "deterministic_flags": {...},
      "summary_effects": {...},
      "top_evidence": [...]
  },

  "warnings": [],
  "missing_reason": "",
}
```

LLM优先只看`decision_metrics`，full metrics通过refs访问。

---

# 68.统计禁区：必须写自动测试

## 禁止1

```text
q > 0.05 -> no biology -> Drop
```

## 禁止2

```text
p > 0.05 -> similar -> Merge
```

## 禁止3

```text
fused network自己产生Split
+
同一个fused network分离好
-> 直接Split
```

必须进一步原始模态/四维审查。

## 禁止4

```text
stage显著相关 -> echo
```

Echo是near-redundant partition。

## 禁止5

```text
scanner p<0.05 -> technical artifact
```

必须有material effect。

## 禁止6

```text
其他set变了 -> 所有旧证据失效
```

必须local signatures。

---

# 69.当前C0001/C0002/C0003应作为regression tests，而非硬编码

## C0003旧结构候选

历史结果大致：

```text
12 + 18
CT q≈0.01
WSI q≈0.01
RNA q≈0.01
Genomic weak
Fused q≈0.005
```

新协议预期：

```text
eligible structural proposal
+
strong cross-modal Split candidate
```

所以：

```text
不能在评估这个proposal之前因为biology identity inconclusive而Drop C0003
```

这是关键regression test。

---

## C0002旧Split

大致：

```text
WSI strong
CT weak
RNA weak
Genomic weak
```

只有1个原始模态支持。

预期：

```text
Split not actionable
```

---

## C0001旧Split

大致：

```text
CT strong
WSI strong
RNA borderline
Genomic weak
```

若最终只是CT+WSI支持：

```text
split_require_molecular_or_biology=true
```

预期先请求：

```text
biological_support / split_proposal
```

即：

```text
child A vs child B
```

不是Split后才用child-vs-rest biology倒推。

---

# 70.最低测试集

## Test A：Clean Accept

输入：

- >=2 modalities identity support；
- no confound；
- no echo；
- biology supporting或inconclusive；
- no actionable proposal。

预期：

```text
accept allowed
```

---

## Test B：Biology inconclusive不等于Drop

输入：

- cross-modal strong；
- biology inconclusive；
- confound clean；
- known-label clean。

预期：

```text
accept or unresolved
```

禁止biology alone Drop。

---

## Test C：Multimodal Split

- eligible fused proposal；
- CT+RNA支持same child labels；
- no confound；
- no echo。

预期：

```text
split allowed
```

---

## Test D：Imaging-only Split

- CT+WSI支持；
- RNA/Genomic弱；
- biology split未attempt。

预期：

```text
need_more_evidence biological_support/split_proposal
```

---

## Test E：Technical Split

- 多模态似乎支持；
- child A/B几乎完全被phase/scanner解释。

预期：

```text
split blocked
```

---

## Test F：Positive Merge

- >=2 modalities median silhouette<=0；
- >=2 modalities median margin<=0；
- no biology distinction veto。

预期：

```text
merge allowed
```

---

## Test G：Nonsignificant不是Merge

- ordinary difference tests p>0.05；
- 但silhouette/margin仍positive。

预期：

```text
merge not supported
```

---

## Test H：Biology veto Merge

- cross-modal somewhat weak；
- strong coherent RNA/CNV distinction。

预期：

```text
merge blocked
```

---

## Test I：Known-label echo

- candidate near-identical to known taxonomy；
- no added multimodal/biology identity；
- no better structural revision。

预期：

```text
drop may be allowed
```

---

## Test J：Evidence lifecycle

- C0001Split；
- C0003membership unchanged。

预期：

```text
C0003 set_identity evidence remains valid
```

partition-level known-label evidence应失效并重算。

---

# 71.临床/生存验证的位置

本轮**不要把clinical relevance加入4个review dimensions**。

最终taxonomy冻结后再做：

- OS；
- 可靠的DFS/PFI/DFI endpoint；
- Kaplan-Meier；
- log-rank；
- univariable Cox；
- multivariable Cox adjusted age/stage/grade；
- stage/grade/age/sex descriptive association。

这些是：

```text
post-discovery clinical validation
```

不得反向用survival p值调Split/Merge或K。

---

# 72.论文中建议如何描述Agent

不要写：

```text
The LLM discovered/split the subtype.
```

推荐：

> Candidate structural revisions were generated deterministically from the fused patient-similarity network. Proposal-specific evidence was quantified across biological, modality-specific representation, technical-confounding and known-taxonomy dimensions using predefined statistical procedures. The language model acted only as a protocol-constrained evidence auditor and action router; statistical metrics, proposal eligibility, action validity and membership updates were computed and enforced deterministically.

核心：

```text
LLM = protocol-constrained evidence reasoning
Python = statistical/execution truth layer
```

---

# 73.重要论文措辞边界

## Cross-modal

由于CT/WSI/RNA/Genomic参与了SNF：

建议写：

```text
modality-specific support
internal multimodal concordance
```

不要写：

```text
independent validation
```

---

## Biology

RNA/WXS/CNV也参与discovery。

建议写：

```text
biological characterization
biological support
```

不要直接写：

```text
independent biological validation
```

---

## External validation

真正external validation需要：

```text
independent cohort
```

不应由本agent内部指标替代。

---

# 74.Codex实施顺序

严格按以下顺序：

### Phase 1
Schema + protocol：

- 4 dimensions；
- scopes；
- proposal_id；
- subject_signature；
- unresolved terminal。

### Phase 2
从现`tool_structural_adequacy.py`抽出Structure Proposal Generator。

### Phase 3
重构evidence registry和local signature。

先通过Evidence lifecycle Test J。

### Phase 4
重构Cross-modal tool三scope。

### Phase 5
重构Biology：
- pathway；
- WXS；
- CNV。

### Phase 6
重构Confounder。

### Phase 7
重构Known-label。

### Phase 8
重写Python action contracts。

### Phase 9
最后修改Verifier/Router/Reviser prompts。

不要先靠prompt去弥补数据模型问题。

### Phase 10
跑全部regression tests。

---

# 75.Definition of Done

- [ ] `structural_adequacy`已从`EVIDENCE_DIMENSIONS`删除；
- [ ] Structure Proposal Generator仍能生成合法Split/Merge proposals；
- [ ] Finding有`scope`；
- [ ] proposal evidence有`proposal_id`；
- [ ] evidence有`subject_signature`；
- [ ] set identity evidence不再绑定whole-partition signature；
- [ ] C0001变化不会让未改变C0003 identity evidence失效；
- [ ] biology inconclusive不会硬Drop；
- [ ] `p>0.05`不会变成Merge support；
- [ ] Split至少有2个原始模态支持；
- [ ] imaging-only Split需要molecular或biology补证；
- [ ] Confounder conflict同时使用significance+effect；
- [ ] Known-label echo是partition redundancy而不是普通association；
- [ ] Drop需要positive invalidating evidence；
- [ ] unresolved是合法terminal；
- [ ] scientific Drop不会被标成runtime failure；
- [ ] CNV进入biological characterization；
- [ ] tool failure不被解释成negative scientific evidence；
- [ ] 所有action引用真实metric refs；
- [ ] LLM不能直接修改membership；
- [ ] Split/Merge只能执行已存在、已验证的legal proposal。

---

# 76.方法学文献依据

1. Wang B, Mezlini AM, Demir F, et al. **Similarity network fusion for aggregating data types on a genomic scale.** Nature Methods. 2014;11:333-337. DOI:10.1038/nmeth.2810.
2. The Cancer Genome Atlas Research Network. **Comprehensive molecular characterization of clear cell renal cell carcinoma.** Nature. 2013;499:43-49. DOI:10.1038/nature12222.
3. Clark DJ, Dhanasekaran SM, Petralia F, et al. **Integrated Proteogenomic Characterization of Clear Cell Renal Cell Carcinoma.** Cell. 2019;179:964-983.e31. DOI:10.1016/j.cell.2019.10.007.
4. Liu Z, Wu Y, Xu H, et al. **Multimodal fusion of radio-pathology and proteogenomics identify integrated glioma subtypes with prognostic and therapeutic opportunities.** Nature Communications. 2025;16:3510. DOI:10.1038/s41467-025-58675-9.
5. Wilkerson MD, Hayes DN. **ConsensusClusterPlus: a class discovery tool with confidence assessments and item tracking.** Bioinformatics. 2010;26:1572-1573. DOI:10.1093/bioinformatics/btq170.
6. John CR, Watson D, Russ D, et al. **M3C: Monte Carlo reference-based consensus clustering.** Scientific Reports. 2020;10:1816. DOI:10.1038/s41598-020-58766-1.
7. Subramanian A, Tamayo P, Mootha VK, et al. **Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles.** PNAS. 2005;102:15545-15550. DOI:10.1073/pnas.0506580102.
8. Hänzelmann S, Castelo R, Guinney J. **GSVA: gene set variation analysis for microarray and RNA-seq data.** BMC Bioinformatics. 2013;14:7. DOI:10.1186/1471-2105-14-7.
9. Mermel CH, Schumacher SE, Hill B, et al. **GISTIC2.0 facilitates sensitive and confident localization of the targets of focal somatic copy-number alteration in human cancers.** Genome Biology. 2011;12:R41. DOI:10.1186/gb-2011-12-4-r41.
10. Rousseeuw PJ. **Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.** Journal of Computational and Applied Mathematics. 1987;20:53-65. DOI:10.1016/0377-0427(87)90125-7.
11. Anderson MJ. **A new method for non-parametric multivariate analysis of variance.** Austral Ecology. 2001;26:32-46. DOI:10.1111/j.1442-9993.2001.01070.pp.x.
12. Vinh NX, Epps J, Bailey J. **Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance.** Journal of Machine Learning Research. 2010;11:2837-2854.
13. Johnson WE, Li C, Rabinovic A. **Adjusting batch effects in microarray expression data using empirical Bayes methods.** Biostatistics. 2007;8:118-127. DOI:10.1093/biostatistics/kxj037.
14. Schuirmann DJ. **A comparison of the Two One-Sided Tests Procedure and the Power Approach for assessing the equivalence of average bioavailability.** Journal of Pharmacokinetics and Biopharmaceutics. 1987;15:657-680. DOI:10.1007/BF01068419.
15. Bergsma W. **A bias-correction for Cramér's V and Tschuprow's T.** Journal of the Korean Statistical Society. 2013;42:323-328. DOI:10.1016/j.jkss.2012.10.002.
16. Cliff N. **Dominance Statistics: Ordinal Analyses to Answer Ordinal Questions.** Psychological Bulletin. 1993;114:494-509. DOI:10.1037/0033-2909.114.3.494.
17. Benjamini Y, Hochberg Y. **Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.** Journal of the Royal Statistical Society Series B. 1995;57:289-300.
18. 正式论文的radiomics technical-confounding部分，另请补充本项目实际采用的scanner/protocol/reconstruction reproducibility原始研究引用；不要只引用review。

---

# 77.给Codex的最终执行指令

> 不要继续在旧`Structural adequacy`validation protocol上打补丁。以本文档为科学规格，将`subtype_review`重构为“4个科学证据维度×scope+独立Structure Proposal Generator+5动作Router”。保留当前上游数据、affinity和candidate partition。所有统计计算、proposal资格、action contract、evidence signature和membership更新必须由确定性Python实现；LLM只负责协议约束下的证据审计、动作选择和supported legal plan选择。任何“不显著”不得自动解释为“相同/应Merge”，任何“biology不显著”不得自动解释为“应Drop”。实现完成后先跑单元/回归测试，再仅重跑`subtype_review`，不要重跑QC、feature engineering或SNF。
