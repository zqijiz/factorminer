# FactorMiner 项目架构与算法详细文档

## 1. 项目整体框架介绍

FactorMiner 是一个由大语言模型 (LLM) 驱动的量化因子挖掘框架。它结合了：
- 基于 OHLCV 市场特征的强类型领域特定语言 (DSL)
- LLM 引导的挖掘循环
- 结构化的经验记忆库
- 基于预测能力和正交性的因子库准入与替换机制
- 严格的运行时重计算，用于分析和基准测试报告
- 扩展的 Helix 阶段 2 研发通道，用于检索、规范化和准入后验证

### 核心架构：
- **双执行通道**：
  - **Paper Lane (RalphLoop)**：严格的，面向基准测试的挖掘循环。用于可复现的论文风格运行、因子库冻结、运行时评估。
  - **Helix Lane (HelixLoop)**：扩展研究模式。支持辩论、知识图谱检索、感知因子家族的提示词、规范化、阶段 2 验证。
- **阶段模型 (Stage Pipeline)**：包含 RetrieveStage, GenerateStage, EvaluateStage, LibraryUpdateStage, DistillStage。
- **内存系统 (Memory Policy)**：支持 paper, none, kg, family_aware, regime_aware 等策略。
- **评估内核 (Evaluation Kernel)**：负责评分、冗余性检查和库几何形状感知评估。

## 2. 具体算法结构与流程

1. **检索阶段 (RetrieveStage)**：根据指定的内存策略，从经验记忆库和知识图谱中检索上下文信息。
2. **生成阶段 (GenerateStage)**：构建提示词上下文 (PromptContextBuilder)，调用 LLM 生成候选因子的 DSL 表达式。
3. **评估阶段 (EvaluateStage)**：解析 DSL 为表达式树，并在给定的数据集上进行评估计算，计算指标（如 IC, 收益率），通过评估内核检查因子有效性和冗余度。
4. **库更新阶段 (LibraryUpdateStage)**：由 FactorAdmissionService 根据准入阈值和依赖/冗余矩阵（如 Spearman, Pearson, Distance Correlation）进行因子替换或新增。
5. **提炼阶段 (DistillStage)**：更新因子生命周期记录，更新内存策略（经验、失败记录、成功记录等），实现自我进化。

## 3. 函数调用关系与模块细节

以下内容通过代码语法树 (AST) 自动解析提取，精确到每个模块、类以及方法/函数之间的调用关系。

### 文件: `factorminer/__init__.py`

**模块说明**: FactorMiner: LLM-powered quantitative factor mining with evolutionary search.

### 文件: `factorminer/agent/__init__.py`

**模块说明**: LLM agent integration for factor generation.

### 文件: `factorminer/agent/critic.py`

**模块说明**: Critic agent that multi-dimensionally scores candidate factors.

The ``CriticAgent`` pre-filters proposals from specialist agents along six
dimensions before any expensive backtesting occurs.  Only the top-scoring
fraction proceeds to IC evaluation, dramatically reducing wasted compute.

Scoring pipeline:
1. Structural heuristics (complexity, operator diversity) -- O(1) per factor.
2. Novelty scoring via string-level edit-distance and token overlap -- O(n).
3. Pattern alignment against success memory -- keyword matching -- O(n).
4. LLM scoring of top candidates for economic intuition -- one API call.
5. Composite score computation and ranking.

#### 类: `CriticScore`

**说明**: Multi-dimensional scored review of a single candidate factor.

##### 方法: `CriticScore.novelty_score`

- **内部/外部调用**: `get`

##### 方法: `CriticScore.quality_score`

- **内部/外部调用**: `get`

##### 方法: `CriticScore.diversity_bonus`

- **内部/外部调用**: `get`

##### 方法: `CriticScore.critic_rationale`


##### 方法: `CriticScore.final_score`


#### 类: `CriticAgent`

**说明**: LLM-powered multi-dimensional critic for candidate factor pre-filtering.

##### 方法: `CriticAgent.__init__`


##### 方法: `CriticAgent.score_batch`

- **说明**: Score a flat list of candidate formula strings.
- **内部/外部调用**: `enumerate`, `setdefault`, `append`, `get`, `_score_proposals`, `_try_build_candidate`

##### 方法: `CriticAgent.review_candidates`

- **说明**: Review all specialist proposals and return ranked scores.
- **内部/外部调用**: `sort`, `normalize_factor_references`, `_memory_signal_to_str`, `get`, `_score_proposals`

##### 方法: `CriticAgent._score_proposals`

- **说明**: Full multi-dimensional scoring pipeline.
- **内部/外部调用**: `_compute_composite`, `max`, `_apply_diversity_adjustment`, `_llm_economic_intuition`, `append`, `CriticScore`, `_brief_heuristic_critique`, `_heuristic_score`, `sort`, `items`, `enumerate`

##### 方法: `CriticAgent._heuristic_score`

- **说明**: Compute heuristic dimension scores without LLM call.
- **内部/外部调用**: `_formula_depth`, `_score_pattern_alignment`, `_score_operator_diversity`, `_score_regime_appropriateness`, `_extract_operators`, `_score_complexity`, `_score_novelty`, `fromkeys`

##### 方法: `CriticAgent._score_novelty`

- **说明**: Novelty: 1.0 = completely novel, 0.0 = exact duplicate.
- **内部/外部调用**: `max`, `_token_idf_similarity`, `min`, `_edit_distance_normalized`, `sum`

##### 方法: `CriticAgent._score_complexity`

- **说明**: Complexity fitness: 1.0 = optimal (depth 3-7, 3-5 unique ops).
- **内部/外部调用**: `max`

##### 方法: `CriticAgent._score_operator_diversity`

- **说明**: Operator diversity: how many distinct operator categories appear?
- **内部/外部调用**: `get`

##### 方法: `CriticAgent._score_pattern_alignment`

- **说明**: Pattern alignment: do formula tokens appear in known success patterns?
- **内部/外部调用**: `lower`, `findall`, `min`

##### 方法: `CriticAgent._score_regime_appropriateness`

- **说明**: Does this formula suit the stated regime context?
- **内部/外部调用**: `any`, `lower`

##### 方法: `CriticAgent._compute_composite`

- **说明**: Compute weighted composite score from dimension scores.
- **内部/外部调用**: `get`, `items`

##### 方法: `CriticAgent._brief_heuristic_critique`

- **说明**: Generate a brief human-readable critique from heuristic scores.
- **内部/外部调用**: `_formula_depth`, `join`, `sorted`, `_extract_operators`, `append`, `get`

##### 方法: `CriticAgent._llm_economic_intuition`

- **说明**: Send top candidates to LLM for economic intuition scoring.
- **内部/外部调用**: `warning`, `_parse_llm_scoring_response`, `_build_llm_scoring_prompt`, `generate`

##### 方法: `CriticAgent._build_llm_scoring_prompt`

- **说明**: Build the structured scoring prompt for LLM economic intuition.
- **内部/外部调用**: `join`, `append`

##### 方法: `CriticAgent._parse_llm_scoring_response`

- **说明**: Parse LLM scoring response into economic intuition scores.
- **内部/外部调用**: `compile`, `max`, `findall`, `debug`, `min`, `get`, `loads`

##### 方法: `CriticAgent._apply_diversity_adjustment`

- **说明**: Slightly boost underrepresented specialists to maintain balance.
- **内部/外部调用**: `values`, `max`, `append`, `Counter`, `min`, `CriticScore`, `sum`, `sort`, `get`

##### 方法: `CriticAgent._memory_signal_to_str`

- **说明**: Flatten a memory signal dict to a compact string for embedding.
- **内部/外部调用**: `join`, `extend`, `append`, `get`

##### 方法: `CriticAgent._fallback_uniform_scores`

- **说明**: Generate uniform scores when all scoring mechanisms fail.
- **内部/外部调用**: `items`, `CriticScore`, `append`

#### 函数: `_extract_operators`

- **说明**: Extract all operator names from a formula string.
- **内部/外部调用**: `findall`

#### 函数: `_formula_depth`

- **说明**: Estimate nesting depth by counting maximum parenthesis depth.
- **内部/外部调用**: `max`

#### 函数: `_tokenize_formula`

- **说明**: Tokenize a formula into its operator and feature tokens.
- **内部/外部调用**: `add`, `update`, `findall`

#### 函数: `_edit_distance_normalized`

- **说明**: Compute normalized edit distance between two formula strings.
- **内部/外部调用**: `max`, `append`, `min`, `range`, `enumerate`

#### 函数: `_token_idf_similarity`

- **说明**: Compute TF-IDF-inspired token overlap similarity.
- **内部/外部调用**: `_tokenize_formula`, `log`, `Counter`, `min`, `sum`

### 文件: `factorminer/agent/debate.py`

**模块说明**: Multi-agent debate orchestrator for factor generation (FactorMAD).

``DebateGenerator`` is a **drop-in replacement** for ``FactorGenerator``.
It runs multiple domain-specialist generators, collects their proposals,
passes them through a multi-dimensional ``CriticAgent`` for pre-filtering,
and returns a single ``List[CandidateFactor]`` with the same interface as
``FactorGenerator.generate_batch()``.

The full pipeline (``DebateOrchestrator``) also supports:
- SymPy-based algebraic deduplication via ``FormulaCanonicalizer``.
- ``DebateMemory`` tracking: specialist leaderboards, blind spot detection.
- Parallel specialist generation (thread-pool).
- Structured ``DebateResult`` dataclass capturing the full debate state.

#### 类: `DebateConfig`

**说明**: Configuration for the multi-agent FactorMAD pipeline.

#### 类: `DebateResult`

**说明**: Full structured result from one debate round.

#### 类: `DebateMemory`

**说明**: Tracks debate history across rounds: who proposed what, what got admitted.

##### 方法: `DebateMemory.__init__`

- **内部/外部调用**:

##### 方法: `DebateMemory.record_round`

- **说明**: Record outcome of one debate round.
- **内部/外部调用**: `setdefault`, `append`, `items`

##### 方法: `DebateMemory.get_specialist_leaderboard`

- **说明**: Return specialist performance sorted by admission rate.
- **内部/外部调用**: `max`, `append`, `sum`, `sort`, `get`

##### 方法: `DebateMemory.get_best_critic_patterns`

- **说明**: Return formula patterns the critic loved that were also admitted.
- **内部/外部调用**:

##### 方法: `DebateMemory.get_blind_spots`

- **说明**: Detect operator families that no specialist is proposing.
- **内部/外部调用**: `values`, `_extract_operators`, `Counter`, `get`

##### 方法: `DebateMemory.get_memory_summary_for_specialist`

- **说明**: Return a brief performance summary for a specific specialist.
- **内部/外部调用**: `sum`, `get`

##### 方法: `DebateMemory.total_rounds`

- **内部/外部调用**:

#### 类: `DebateOrchestrator`

**说明**: Orchestrates the full multi-agent FactorMAD debate cycle.

##### 方法: `DebateOrchestrator.__init__`


##### 方法: `DebateOrchestrator.run_debate_round`

- **说明**: Run one full debate round and return structured results.
- **内部/外部调用**: `DebateResult`, `get`, `lower`, `setdefault`, `items`, `append`, `_deduplicate`, `generate_proposals`, `_generate_parallel`, `_flatten_memory_signal`, `info`, `normalize_factor_references`, `_score_proposals`, `_try_build_candidate`

##### 方法: `DebateOrchestrator._generate_parallel`

- **说明**: Generate from all specialists concurrently using a thread pool.
- **内部/外部调用**: `ThreadPoolExecutor`, `warning`, `submit`, `generate_proposals`, `min`, `as_completed`, `info`, `result`

##### 方法: `DebateOrchestrator._deduplicate`

- **说明**: Remove algebraic duplicates using SymPy canonicalizer if available.
- **内部/外部调用**: `add`, `append`, `try_parse`, `canonicalize`

#### 类: `DebateGenerator`

**说明**: Multi-agent debate-based factor generator (drop-in for FactorGenerator).

##### 方法: `DebateGenerator.__init__`

- **内部/外部调用**: `warning`, `CriticAgent`, `FactorGenerator`, `DebateOrchestrator`, `SpecialistPromptBuilder`, `append`, `FormulaCanonicalizer`, `SpecialistAgent`, `DebateConfig`, `DebateMemory`

##### 方法: `DebateGenerator.generate_batch`

- **说明**: Generate a batch of candidate factors via multi-agent debate.
- **内部/外部调用**: `DebateResult`, `_tag_specialist_source_from_agents`, `get`, `run_debate_round`, `info`, `record_round`, `add`, `values`, `items`, `append`, `generate_batch`, `min`, `_debate_result_to_candidates`, `normalize_factor_references`

##### 方法: `DebateGenerator.last_debate_result`

- **说明**: The ``DebateResult`` from the most recent ``generate_batch`` call.

##### 方法: `DebateGenerator.debate_memory`

- **说明**: The ``DebateMemory`` tracking history across rounds.

##### 方法: `DebateGenerator.get_specialist_leaderboard`

- **说明**: Return specialist admission leaderboard if memory is enabled.
- **内部/外部调用**: `get_specialist_leaderboard`

##### 方法: `DebateGenerator.get_blind_spots`

- **说明**: Return operator family blind spots if memory is enabled.
- **内部/外部调用**: `get_blind_spots`

##### 方法: `DebateGenerator.update_specialist_admissions`

- **说明**: Feed evaluation results back to specialist agents and debate memory.
- **内部/外部调用**: `record_round`, `update_domain_memory`, `append`, `index`, `get`

##### 方法: `DebateGenerator._debate_result_to_candidates`

- **说明**: Convert DebateResult critic scores into CandidateFactor objects.
- **内部/外部调用**: `add`, `append`, `sort`, `_try_build_candidate`

##### 方法: `DebateGenerator._tag_specialist_source_from_agents`

- **说明**: Tag candidate source if not already embedded in category.
- **内部/外部调用**: `items`, `startswith`

##### 方法: `DebateGenerator._scores_to_candidates`

- **说明**: Map CriticScore objects back to CandidateFactor instances.
- **内部/外部调用**: `values`, `add`, `append`, `get`

##### 方法: `DebateGenerator._tag_specialist_source`

- **说明**: Add specialist source information to each candidate's category.
- **内部/外部调用**: `get`, `items`, `startswith`

#### 函数: `_flatten_memory_signal`

- **说明**: Flatten a memory signal dict to a compact string.
- **内部/外部调用**: `join`, `extend`, `append`, `get`

### 文件: `factorminer/agent/factor_generator.py`

**模块说明**: Main factor generation agent using LLM guided by memory priors.

Orchestrates the prompt construction, LLM invocation, output parsing,
and retry logic for a single batch of factor candidates.

#### 类: `FactorGenerator`

**说明**: LLM-based factor generation agent.

##### 方法: `FactorGenerator.__init__`

- **内部/外部调用**: `PromptBuilder`

##### 方法: `FactorGenerator.generate_batch`

- **说明**: Generate a batch of candidate factors using LLM guided by memory priors.
- **内部/外部调用**: `add`, `monotonic`, `build_user_prompt`, `join`, `parse_llm_output`, `sorted`, `generate`, `append`, `_retry_failed_parses`, `items`, `info`, `get`

##### 方法: `FactorGenerator._retry_failed_parses`

- **说明**: Retry parsing failed outputs with a repair prompt.
- **内部/外部调用**: `warning`, `max`, `join`, `debug`, `parse_llm_output`, `extend`, `generate`, `range`, `enumerate`

### 文件: `factorminer/agent/llm_interface.py`

**模块说明**: Abstract LLM interface supporting multiple providers.

Provides a unified API for generating text completions across OpenAI,
Anthropic, Google (Gemini), and a deterministic mock provider for testing.

#### 类: `MissingAPIKeyError`

**说明**: Raised when a non-mock provider is used without a configured API key.

#### 类: `LLMProvider`

**说明**: Abstract base for LLM text-generation providers.

##### 方法: `LLMProvider.generate`

- **说明**: Generate a text completion.

##### 方法: `LLMProvider.provider_name`

- **说明**: Human-readable provider name.

#### 类: `OpenAIProvider`

**说明**: OpenAI API provider (GPT-4, GPT-4o, etc.).

##### 方法: `OpenAIProvider.__init__`

- **内部/外部调用**: `get`

##### 方法: `OpenAIProvider._get_client`

- **内部/外部调用**: `OpenAI`, `ImportError`, `MissingAPIKeyError`

##### 方法: `OpenAIProvider.generate`

- **内部/外部调用**: `debug`, `create`, `_get_client`

##### 方法: `OpenAIProvider.provider_name`


#### 类: `AnthropicProvider`

**说明**: Anthropic Claude API provider with adaptive thinking support.

##### 方法: `AnthropicProvider.__init__`

- **内部/外部调用**: `get`

##### 方法: `AnthropicProvider._get_client`

- **内部/外部调用**: `Anthropic`, `ImportError`, `MissingAPIKeyError`

##### 方法: `AnthropicProvider.generate`

- **内部/外部调用**: `create`, `join`, `debug`, `_get_client`, `append`

##### 方法: `AnthropicProvider.provider_name`


#### 类: `GoogleProvider`

**说明**: Google Gemini API provider (paper uses Gemini 3.0 Flash).

##### 方法: `GoogleProvider.__init__`

- **内部/外部调用**: `get`

##### 方法: `GoogleProvider._get_client`

- **内部/外部调用**: `ImportError`, `configure`, `GenerativeModel`, `MissingAPIKeyError`

##### 方法: `GoogleProvider.generate`

- **内部/外部调用**: `debug`, `_get_client`, `generate_content`

##### 方法: `GoogleProvider.provider_name`


#### 类: `MockProvider`

**说明**: Deterministic provider for testing without API calls.

##### 方法: `MockProvider.__init__`


##### 方法: `MockProvider.generate`

- **内部/外部调用**: `split`, `join`, `lower`, `append`, `isdigit`, `min`, `range`, `enumerate`

##### 方法: `MockProvider.provider_name`


#### 函数: `create_provider`

- **说明**: Factory function to instantiate an LLM provider from config.
- **内部/外部调用**: `cls`, `sorted`, `keys`, `info`, `get`

### 文件: `factorminer/agent/output_parser.py`

**模块说明**: Parse LLM output into structured CandidateFactor objects.

Handles various output formats from LLMs: numbered lists, JSON,
markdown code blocks, and raw text.  Validates each formula against
the expression tree parser.

#### 类: `CandidateFactor`

**说明**: A candidate factor parsed from LLM output.

##### 方法: `CandidateFactor.is_valid`


#### 函数: `_infer_category`

- **说明**: Infer a rough category from the outermost operators in the formula.
- **内部/外部调用**: `lower`, `any`

#### 函数: `_strip_markdown`

- **说明**: Remove markdown code block markers.
- **内部/外部调用**: `sub`

#### 函数: `_clean_formula`

- **说明**: Clean up a formula string before parsing.
- **内部/外部调用**: `strip`, `rstrip`, `index`

#### 函数: `parse_llm_output`

- **说明**: Parse raw LLM text output into candidate factors.
- **内部/外部调用**: `_generate_name_from_formula`, `lower`, `_strip_markdown`, `any`, `startswith`, `match`, `split`, `append`, `add`, `findall`, `debug`, `sum`, `_clean_formula`, `replace`, `strip`, `isalpha`, `group`, `_try_build_candidate`

#### 函数: `_try_build_candidate`

- **说明**: Attempt to parse a formula and build a CandidateFactor.
- **内部/外部调用**: `to_string`, `CandidateFactor`, `try_parse`, `parse`, `_infer_category`

#### 函数: `_generate_name_from_formula`

- **说明**: Generate a descriptive name from a formula.
- **内部/外部调用**: `lower`, `match`, `group`

### 文件: `factorminer/agent/prompt_builder.py`

**模块说明**: Build prompts for LLM-driven factor generation using memory priors.

The system prompt encodes the full operator library, syntax rules, feature
list, and task description.  The user prompt injects per-iteration context:
memory signals, library state, and output format instructions.

#### 类: `PromptBuilder`

**说明**: Constructs system and user prompts for factor generation.

##### 方法: `PromptBuilder.__init__`


##### 方法: `PromptBuilder.system_prompt`


##### 方法: `PromptBuilder.build_user_prompt`

- **说明**: Build the per-iteration user prompt injecting memory priors.
- **内部/外部调用**: `normalize_factor_references`, `join`, `append`, `items`, `strip`, `get`

#### 函数: `_format_operator_table`

- **说明**: Build a human-readable operator reference table grouped by category.
- **内部/外部调用**: `values`, `join`, `sorted`, `setdefault`, `append`, `range`, `get`

#### 函数: `_format_feature_list`

- **说明**: Build a description of available raw features.
- **内部/外部调用**: `join`, `get`, `append`

#### 函数: `normalize_factor_references`

- **说明**: Convert mixed factor metadata into prompt-safe string references.
- **内部/外部调用**: `add`, `append`, `strip`, `get`

#### 函数: `build_specialist_prompt`

- **说明**: Build a rich context-aware user prompt for a specialist agent.
- **内部/外部调用**: `normalize_factor_references`, `join`, `append`, `items`, `strip`, `get`

#### 函数: `build_critic_scoring_prompt`

- **说明**: Build a structured JSON-output scoring prompt for the critic agent.
- **内部/外部调用**: `join`, `get`, `append`

#### 函数: `build_debate_synthesis_prompt`

- **说明**: Build a consensus synthesis prompt for the debate orchestrator.
- **内部/外部调用**: `join`, `sorted`, `get`, `append`

### 文件: `factorminer/agent/specialists.py`

**模块说明**: Specialist agent configurations for domain-focused factor generation.

Each specialist focuses on a particular alpha factor domain with a distinct
cognitive style, preferred operators, domain hypotheses, and historical
success tracking.  ``SpecialistAgent`` wraps a config with per-domain memory
and proposal logic.  ``SpecialistPromptBuilder`` extends the base
``PromptBuilder`` to inject domain-specific directives.

#### 类: `SpecialistConfig`

**说明**: Configuration for a domain-specialist factor generator.

#### 类: `SpecialistDomainMemory`

**说明**: Tracks admission/rejection history for a single specialist.

##### 方法: `SpecialistDomainMemory.total_proposed`

- **内部/外部调用**:

##### 方法: `SpecialistDomainMemory.success_rate`

- **内部/外部调用**:

##### 方法: `SpecialistDomainMemory.record_admitted`

- **内部/外部调用**: `extend`

##### 方法: `SpecialistDomainMemory.record_rejected`

- **内部/外部调用**: `extend`

##### 方法: `SpecialistDomainMemory.get_summary`

- **说明**: Human-readable summary of domain performance.
- **内部/外部调用**: `join`, `append`, `most_common`, `Counter`

#### 类: `SpecialistAgent`

**说明**: Domain-specialist factor proposer with memory and success tracking.

##### 方法: `SpecialistAgent.__init__`

- **内部/外部调用**: `SpecialistPromptBuilder`, `SpecialistDomainMemory`

##### 方法: `SpecialistAgent.name`


##### 方法: `SpecialistAgent.success_rate`

- **说明**: Fraction of this specialist's proposals that were admitted.

##### 方法: `SpecialistAgent.generate_proposals`

- **说明**: Generate formula string proposals from this specialist.
- **内部/外部调用**: `warning`, `build_user_prompt`, `debug`, `_enrich_memory_signal`, `parse_llm_output`, `generate`, `normalize_factor_references`

##### 方法: `SpecialistAgent.update_domain_memory`

- **说明**: Update this specialist's domain memory after evaluation.
- **内部/外部调用**: `record_rejected`, `record_admitted`

##### 方法: `SpecialistAgent.get_domain_performance_summary`

- **说明**: Human-readable summary of what this specialist has discovered.
- **内部/外部调用**: `get_summary`

##### 方法: `SpecialistAgent._enrich_memory_signal`

- **说明**: Merge base memory signal with domain-specific context.
- **内部/外部调用**: `join`, `append`, `get`

#### 类: `SpecialistPromptBuilder`

**说明**: Prompt builder that injects domain-specific specialist directives.

##### 方法: `SpecialistPromptBuilder.__init__`

- **内部/外部调用**: `__init__`

##### 方法: `SpecialistPromptBuilder.specialist_config`

- **说明**: Return the underlying specialist configuration.

##### 方法: `SpecialistPromptBuilder.build_user_prompt`

- **说明**: Build user prompt with specialist operator/feature bias.
- **内部/外部调用**: `join`, `build_user_prompt`

### 文件: `factorminer/architecture/__init__.py`

**模块说明**: Canonical protocol and runtime architecture contracts for FactorMiner.

#### 函数: `__getattr__`

- **内部/外部调用**: `AttributeError`, `import_module`, `globals`

### 文件: `factorminer/architecture/dataset_contract.py`

**模块说明**: Stable dataset pipeline contracts for mining and analysis.

#### 类: `DatasetContract`

**说明**: Canonical description of how raw data became the mining tensors.

##### 方法: `DatasetContract.from_runtime_dataset`

- **内部/外部调用**: `cls`, `max`, `_safe_len`, `keys`, `shape`, `items`

##### 方法: `DatasetContract.from_arrays`

- **内部/外部调用**: `cls`, `keys`, `shape`, `ndim`

##### 方法: `DatasetContract.to_dict`

- **内部/外部调用**: `asdict`

#### 函数: `_safe_len`

- **内部/外部调用**:

### 文件: `factorminer/architecture/dependence.py`

**模块说明**: Pluggable dependence metrics for library redundancy and replacement logic.

#### 类: `DependenceMetric`

**说明**: Abstract pairwise dependence metric over cross-sectional signals.

##### 方法: `DependenceMetric.compute`


##### 方法: `DependenceMetric.describe`


#### 类: `SpearmanDependenceMetric`

**说明**: Mean absolute Spearman rank correlation across time.

##### 方法: `SpearmanDependenceMetric.compute`

- **内部/外部调用**: `rankdata`, `mean`, `append`, `_pearson_abs`, `_iter_valid_columns`

#### 类: `PearsonDependenceMetric`

**说明**: Mean absolute Pearson correlation across time.

##### 方法: `PearsonDependenceMetric.compute`

- **内部/外部调用**: `mean`, `append`, `_pearson_abs`, `_iter_valid_columns`

#### 类: `DistanceCorrelationMetric`

**说明**: Mean distance-correlation dependence across time.

##### 方法: `DistanceCorrelationMetric.compute`

- **内部/外部调用**: `_distance_correlation_abs`, `mean`, `append`, `_iter_valid_columns`

#### 函数: `_iter_valid_columns`

- **内部/外部调用**: `range`, `sum`, `isnan`

#### 函数: `_pearson_abs`

- **内部/外部调用**: `mean`, `abs`, `sqrt`, `sum`

#### 函数: `_distance_correlation_abs`

- **内部/外部调用**: `max`, `reshape`, `mean`, `abs`, `sqrt`, `asarray`

#### 函数: `build_dependence_metric`

- **说明**: Build one supported dependence metric from config/runtime names.
- **内部/外部调用**: `SpearmanDependenceMetric`, `DistanceCorrelationMetric`, `lower`, `PearsonDependenceMetric`, `strip`

### 文件: `factorminer/architecture/evaluation_kernel.py`

**模块说明**: Shared evaluation kernel for signals, metrics, admission, and dedup.

#### 类: `EvaluationKernel`

**说明**: Canonical evaluation engine used by loops and runtime analysis.

##### 方法: `EvaluationKernel.build_data_dict`

- **内部/外部调用**: `enumerate`

##### 方法: `EvaluationKernel.compute_signals`

- **内部/外部调用**: `try_parse`, `asarray`, `compute_tree_signals`, `SignalComputationError`

##### 方法: `EvaluationKernel.compute_target_stats`

- **内部/外部调用**: `compute_factor_stats`, `items`

##### 方法: `EvaluationKernel.compute_quality_score`

- **内部/外部调用**: `values`, `next`, `iter`, `to_dict`, `build_score_vector`, `compute_factor_geometry`, `candidate_geometry`, `passes_research_admission`, `get`

##### 方法: `EvaluationKernel.admission_decision`

- **内部/外部调用**: `check_admission`

##### 方法: `EvaluationKernel.replacement_decision`

- **内部/外部调用**: `check_replacement`

##### 方法: `EvaluationKernel.deduplicate_results`

- **内部/外部调用**: `compute_correlation`, `sorted`, `append`, `min`, `enumerate`

### 文件: `factorminer/architecture/families.py`

**模块说明**: Factor-family discovery and prompt-facing family diagnostics.

#### 类: `FactorFamily`

##### 方法: `FactorFamily.to_dict`

- **内部/外部调用**: `asdict`

#### 类: `FactorFamilyDiscovery`

**说明**: Discover family structure and prompt-facing gaps from formulas/library state.

##### 方法: `FactorFamilyDiscovery.summarize`

- **内部/外部调用**: `_underexplored_families`, `extend`, `_recommended_families`, `to_dict`, `discover`, `_saturated_families`, `_prompt_text`, `get`

##### 方法: `FactorFamilyDiscovery.discover`

- **内部/外部调用**: `values`, `get`, `extract_operators`, `sorted`, `setdefault`, `FactorFamily`, `append`, `infer_family`, `items`, `extract_features`

##### 方法: `FactorFamilyDiscovery._saturated_families`

- **内部/外部调用**: `values`, `max`, `sorted`, `items`, `sum`, `get`

##### 方法: `FactorFamilyDiscovery._underexplored_families`

- **内部/外部调用**: `_recommended_families`, `sorted`

##### 方法: `FactorFamilyDiscovery._recommended_families`

- **内部/外部调用**: `add`, `sorted`, `infer_family`, `get`

##### 方法: `FactorFamilyDiscovery._prompt_text`

- **内部/外部调用**: `join`, `append`

#### 函数: `extract_operators`

- **内部/外部调用**: `findall`

#### 函数: `extract_features`

- **内部/外部调用**: `findall`

#### 函数: `infer_family`

- **说明**: Infer a stable factor family from a formula string.
- **内部/外部调用**: `extract_operators`, `upper`

### 文件: `factorminer/architecture/geometry.py`

**模块说明**: Reusable library geometry diagnostics and novelty utilities.

#### 类: `CandidateGeometry`

**说明**: Geometric view of one candidate relative to the current library.

#### 类: `LibraryGeometry`

**说明**: Centralizes correlation and saturation geometry around the library.

##### 方法: `LibraryGeometry.__init__`


##### 方法: `LibraryGeometry.candidate_geometry`

- **内部/外部调用**: `list_factors`, `max`, `compute_correlation`, `append`, `CandidateGeometry`

##### 方法: `LibraryGeometry.replacement_target`

- **内部/外部调用**: `candidate_geometry`

##### 方法: `LibraryGeometry.check_admission`

- **内部/外部调用**: `candidate_geometry`

##### 方法: `LibraryGeometry.check_replacement`

- **内部/外部调用**: `get_factor`, `candidate_geometry`

##### 方法: `LibraryGeometry.library_snapshot`

- **内部/外部调用**: `get_diagnostics`, `get`

### 文件: `factorminer/architecture/library_services.py`

**模块说明**: Reusable services for factor-library mutations.

#### 类: `FactorAdmissionService`

**说明**: Owns conversion from evaluation results into library mutations.

##### 方法: `FactorAdmissionService.__init__`


##### 方法: `FactorAdmissionService.admit_results`

- **内部/外部调用**: `warning`, `admit_factor`, `replace_factor`, `append`, `_build_factor`, `info`

##### 方法: `FactorAdmissionService._build_factor`

- **内部/外部调用**: `Factor`, `infer_family`

### 文件: `factorminer/architecture/lifecycle.py`

**模块说明**: Factor candidate lifecycle logging and trajectory reconstruction.

#### 类: `FactorLifecycleEvent`

**说明**: One candidate event emitted during the mining loop.

#### 类: `FactorLifecycleStore`

**说明**: Structured event log for candidate trajectories across iterations.

##### 方法: `FactorLifecycleStore.__init__`

- **内部/外部调用**: `mkdir`, `Path`

##### 方法: `FactorLifecycleStore.record`

- **内部/外部调用**: `asdict`, `open`, `FactorLifecycleEvent`, `append`, `write`, `dumps`

##### 方法: `FactorLifecycleStore.record_batch_results`

- **内部/外部调用**: `record`

##### 方法: `FactorLifecycleStore.record_memory_distillation`

- **内部/外部调用**: `get`, `record`

##### 方法: `FactorLifecycleStore.build_trajectory`

- **内部/外部调用**: `values`, `setdefault`, `get`

### 文件: `factorminer/architecture/memory_policy.py`

**模块说明**: Formal policy boundary for memory retrieval, formation, and persistence.

#### 类: `MemoryPolicy`

**说明**: Policy interface for memory state, retrieval, and evolution.

##### 方法: `MemoryPolicy.schema`


##### 方法: `MemoryPolicy.retrieve`


##### 方法: `MemoryPolicy.form`


##### 方法: `MemoryPolicy.evolve`


##### 方法: `MemoryPolicy.serialize`


##### 方法: `MemoryPolicy.restore`


#### 类: `PaperMemoryPolicy`

**说明**: Default paper-faithful memory policy using the F/E/R operators.

##### 方法: `PaperMemoryPolicy.__init__`


##### 方法: `PaperMemoryPolicy.schema`


##### 方法: `PaperMemoryPolicy.retrieve`

- **内部/外部调用**: `retrieve_memory`, `min`, `schema`

##### 方法: `PaperMemoryPolicy.form`

- **内部/外部调用**: `form_memory`

##### 方法: `PaperMemoryPolicy.evolve`

- **内部/外部调用**: `evolve_memory`

##### 方法: `PaperMemoryPolicy.serialize`

- **内部/外部调用**: `to_dict`, `schema`

##### 方法: `PaperMemoryPolicy.restore`

- **内部/外部调用**: `from_dict`

#### 类: `NoMemoryPolicy`

**说明**: Ablation policy that disables retrieval and distillation.

##### 方法: `NoMemoryPolicy.__init__`


##### 方法: `NoMemoryPolicy.schema`


##### 方法: `NoMemoryPolicy.retrieve`

- **内部/外部调用**: `get`, `schema`

##### 方法: `NoMemoryPolicy.form`


##### 方法: `NoMemoryPolicy.evolve`


##### 方法: `NoMemoryPolicy.serialize`

- **内部/外部调用**: `to_dict`, `schema`

##### 方法: `NoMemoryPolicy.restore`

- **内部/外部调用**: `from_dict`

#### 类: `RegimeAwareMemoryPolicy`

**说明**: Paper memory with regime-conditioned retrieval context and ranking.

##### 方法: `RegimeAwareMemoryPolicy.__init__`

- **内部/外部调用**: `__init__`, `max`, `RegimeConfig`, `classify`, `RegimeDetector`

##### 方法: `RegimeAwareMemoryPolicy.schema`

- **内部/外部调用**: `schema`

##### 方法: `RegimeAwareMemoryPolicy.retrieve`

- **内部/外部调用**: `_bias_score`, `_regime_context`, `sorted`, `retrieve`, `strip`

##### 方法: `RegimeAwareMemoryPolicy._regime_context`

- **内部/外部调用**: `mean`, `MarketRegime`, `min`, `get`

##### 方法: `RegimeAwareMemoryPolicy._bias_score`

- **内部/外部调用**: `join`, `lower`, `sum`, `get`

#### 类: `KGMemoryPolicy`

**说明**: Paper memory augmented with a persistent factor knowledge graph.

##### 方法: `KGMemoryPolicy.__init__`

- **内部/外部调用**: `FactorKnowledgeGraph`, `__init__`

##### 方法: `KGMemoryPolicy.schema`

- **内部/外部调用**: `get_edge_count`, `get_factor_count`, `schema`

##### 方法: `KGMemoryPolicy.retrieve`

- **内部/外部调用**: `get_factor_count`, `schema`, `min`, `retrieve_memory_enhanced`, `get_edge_count`

##### 方法: `KGMemoryPolicy.form`

- **内部/外部调用**: `_update_knowledge_graph`, `form`

##### 方法: `KGMemoryPolicy.serialize`

- **内部/外部调用**: `to_dict`, `serialize`

##### 方法: `KGMemoryPolicy.restore`

- **内部/外部调用**: `get`, `from_dict`

##### 方法: `KGMemoryPolicy._update_knowledge_graph`

- **内部/外部调用**: `add`, `FactorNode`, `add_correlation_edge`, `add_factor`, `list_factor_nodes`, `min`, `infer_family`, `get`

#### 类: `FamilyAwareMemoryPolicy`

**说明**: Paper memory reranked by family saturation and family gaps.

##### 方法: `FamilyAwareMemoryPolicy.__init__`

- **内部/外部调用**: `__init__`, `FactorFamilyDiscovery`

##### 方法: `FamilyAwareMemoryPolicy.schema`

- **内部/外部调用**: `schema`

##### 方法: `FamilyAwareMemoryPolicy.retrieve`

- **内部/外部调用**: `_family_bias`, `sorted`, `retrieve`, `strip`, `summarize`, `get`

##### 方法: `FamilyAwareMemoryPolicy._family_bias`

- **内部/外部调用**: `infer_family`, `get`

#### 函数: `build_memory_policy`

- **说明**: Construct the configured memory policy from flat or hierarchical config.
- **内部/外部调用**: `RegimeAwareMemoryPolicy`, `lower`, `NoMemoryPolicy`, `PaperMemoryPolicy`, `strip`, `FamilyAwareMemoryPolicy`, `KGMemoryPolicy`

### 文件: `factorminer/architecture/paper_protocol.py`

**模块说明**: Paper-faithful benchmark and mining protocol contracts.

#### 类: `TargetDefinition`

**说明**: One named return target in the paper/runtime contract.

#### 类: `ArtifactSchema`

**说明**: Artifact inventory expected from one mining or benchmark run.

#### 类: `TrainTestProtocol`

**说明**: Temporal protocol and Top-K freeze contract.

#### 类: `PaperProtocol`

**说明**: Single object describing the paper-faithful runtime contract.

##### 方法: `PaperProtocol.from_config`

- **内部/外部调用**: `TargetDefinition`, `cls`, `TrainTestProtocol`, `get`

##### 方法: `PaperProtocol.target_stack`

- **内部/外部调用**: `keys`

##### 方法: `PaperProtocol.admission_contract`


##### 方法: `PaperProtocol.replacement_contract`


##### 方法: `PaperProtocol.runtime_contract`

- **内部/外部调用**: `items`, `replacement_contract`, `asdict`, `admission_contract`

### 文件: `factorminer/architecture/phase2_services.py`

**模块说明**: Reusable Phase 2 services extracted from Helix loop logic.

#### 类: `OnlineForgettingService`

**说明**: Applies online forgetting to memory state after dry spells.

##### 方法: `OnlineForgettingService.apply`

- **内部/外部调用**:

#### 类: `KnowledgeGraphService`

**说明**: Owns factor-node construction, correlation edges, and derivation edges.

##### 方法: `KnowledgeGraphService.add_result`

- **内部/外部调用**: `extract_operators`, `FactorNode`, `list_factors`, `compute_correlation`, `add_factor`, `embed`, `add_correlation_edge`, `_detect_derivation`, `infer_family`, `extract_features`

##### 方法: `KnowledgeGraphService.remove_factor`

- **内部/外部调用**: `remove`, `remove_factor`

##### 方法: `KnowledgeGraphService._detect_derivation`

- **内部/外部调用**: `extract_operators`, `max`, `list_factors`, `join`, `sorted`, `add_derivation_edge`

### 文件: `factorminer/architecture/prompt_context.py`

**模块说明**: Boundary from structured memory state to generation prompt context.

#### 类: `PromptContextBuilder`

**说明**: Builds the prompt-facing context payload for factor generation.

##### 方法: `PromptContextBuilder.build`

- **内部/外部调用**: `update`, `runtime_contract`, `summarize`, `get`

### 文件: `factorminer/architecture/research_extensions.py`

**模块说明**: High-value research extension services for family, regime, dependence, and utility.

#### 类: `FamilySummary`

**说明**: Per-family summary with admission and rejection counts.

##### 方法: `FamilySummary.to_dict`

- **内部/外部调用**: `asdict`

#### 类: `FamilyContextSummary`

**说明**: Structured family context for prompt construction.

##### 方法: `FamilyContextSummary.to_dict`

- **内部/外部调用**: `asdict`

#### 类: `RegimeMemorySummary`

**说明**: Structured regime-aware memory summary.

##### 方法: `RegimeMemorySummary.to_dict`

- **内部/外部调用**: `asdict`

#### 类: `DependenceProfile`

**说明**: Nonlinear dependence profile between two signal panels.

##### 方法: `DependenceProfile.to_dict`

- **内部/外部调用**: `asdict`

#### 类: `EnsembleUtilitySummary`

**说明**: Marginal utility estimate for a candidate signal against an ensemble.

##### 方法: `EnsembleUtilitySummary.to_dict`

- **内部/外部调用**: `asdict`

#### 类: `FamilyContextService`

**说明**: Derive factor-family context from admitted and rejected formulas.

##### 方法: `FamilyContextService.__init__`

- **内部/外部调用**: `FactorFamilyDiscovery`

##### 方法: `FamilyContextService.summarize`

- **内部/外部调用**: `any`, `_underexplored_families`, `append`, `max`, `sorted`, `FamilyContextSummary`, `infer_family`, `_normalize_formula_entry`, `items`, `get`, `_recommended_families`, `discover`, `FamilySummary`, `_saturated_families`, `_prompt_text`, `sort`

##### 方法: `FamilyContextService._saturated_families`

- **内部/外部调用**: `values`, `max`, `sorted`, `items`, `sum`, `get`

##### 方法: `FamilyContextService._recommended_families`

- **内部/外部调用**: `add`, `sorted`, `infer_family`, `get`

##### 方法: `FamilyContextService._underexplored_families`

- **内部/外部调用**: `sorted`

##### 方法: `FamilyContextService._prompt_text`

- **内部/外部调用**: `join`, `append`

#### 类: `RegimeMemoryService`

**说明**: Summarize regime labels and metrics into prompt-facing memory context.

##### 方法: `RegimeMemoryService.summarize`

- **内部/外部调用**: `values`, `get`, `max`, `lower`, `setdefault`, `_prompt_text`, `strip`, `_summarize_metric_by_regime`, `count`, `sum`, `RegimeMemorySummary`, `items`, `zip`

##### 方法: `RegimeMemoryService._summarize_metric_by_regime`

- **内部/外部调用**: `ravel`, `mean`, `lower`, `sorted`, `_to_float_array`, `std`, `strip`, `asarray`, `items`

##### 方法: `RegimeMemoryService._prompt_text`

- **内部/外部调用**: `max`, `join`, `sorted`, `append`, `count`, `items`

#### 类: `NonlinearDependenceService`

**说明**: Score nonlinear dependence with distance correlation and an MI proxy.

##### 方法: `NonlinearDependenceService.score_pair`

- **内部/外部调用**: `DependenceProfile`, `_periodwise_nonlinear_scores`

##### 方法: `NonlinearDependenceService.score_collection`

- **内部/外部调用**: `_stack_reference_collection`, `max`, `mean`, `to_dict`, `range`, `score_pair`, `zip`

#### 类: `EnsembleMarginalUtilityService`

**说明**: Estimate candidate utility against the current ensemble and returns.

##### 方法: `EnsembleMarginalUtilityService.__init__`

- **内部/外部调用**: `NonlinearDependenceService`

##### 方法: `EnsembleMarginalUtilityService.estimate`

- **内部/外部调用**: `min`, `_stack_reference_collection`, `_linear_fit_predict`, `empty`, `_safe_corr`, `full`, `EnsembleUtilitySummary`, `_as_panel`, `max`, `score_collection`, `_flatten_period_samples`, `mean`, `_safe_r2`, `round`, `range`, `_prompt_text`

##### 方法: `EnsembleMarginalUtilityService._prompt_text`

- **内部/外部调用**: `join`, `append`

#### 类: `ResearchExtensionService`

**说明**: Composes the research extension services into a single prompt-facing API.

##### 方法: `ResearchExtensionService.__init__`

- **内部/外部调用**: `FamilyContextService`, `RegimeMemoryService`, `EnsembleMarginalUtilityService`, `NonlinearDependenceService`

##### 方法: `ResearchExtensionService.family_context`

- **内部/外部调用**: `summarize`

##### 方法: `ResearchExtensionService.regime_context`

- **内部/外部调用**: `summarize`

##### 方法: `ResearchExtensionService.dependence_profile`

- **内部/外部调用**: `score_pair`

##### 方法: `ResearchExtensionService.ensemble_utility`

- **内部/外部调用**: `estimate`

##### 方法: `ResearchExtensionService.build_prompt_context_extensions`

- **内部/外部调用**: `ensemble_utility`, `regime_context`, `asdict`, `family_context`, `join`, `score_collection`, `append`

#### 函数: `_to_float_array`

- **内部/外部调用**: `asarray`, `reshape`

#### 函数: `_as_panel`

- **内部/外部调用**: `asarray`

#### 函数: `_safe_nanmean`

- **内部/外部调用**: `isfinite`, `mean`

#### 函数: `_safe_corr`

- **内部/外部调用**: `mean`, `isfinite`, `sqrt`, `sum`, `ravel`, `asarray`

#### 函数: `_safe_r2`

- **内部/外部调用**: `mean`, `isfinite`, `sum`, `ravel`, `asarray`

#### 函数: `_standardize_train_test`

- **内部/外部调用**: `mean`, `std`

#### 函数: `_stack_reference_collection`

- **内部/外部调用**: `_as_panel`, `range`

#### 函数: `_flatten_period_samples`

- **内部/外部调用**: `sum`, `where`, `isfinite`, `append`, `empty`, `asarray`

#### 函数: `_linear_fit_predict`

- **内部/外部调用**: `mean`, `_standardize_train_test`, `ones`, `full`, `lstsq`, `column_stack`

#### 函数: `_gaussian_copula_mi_proxy`

- **内部/外部调用**: `rankdata`, `max`, `abs`, `log`, `min`, `_safe_corr`

#### 函数: `_mutual_information_proxy`

- **内部/外部调用**: `rankdata`, `max`, `reshape`, `isfinite`, `mutual_info_regression`, `min`, `_gaussian_copula_mi_proxy`, `sum`, `ravel`, `asarray`

#### 函数: `_periodwise_nonlinear_scores`

- **内部/外部调用**: `_as_panel`, `rankdata`, `max`, `DistanceCorrelationMetric`, `reshape`, `isfinite`, `abs`, `_mutual_information_proxy`, `append`, `_safe_nanmean`, `compute`, `_safe_corr`, `range`, `sum`, `exp`

#### 函数: `_normalize_formula_entry`

- **内部/外部调用**: `strip`, `infer_family`, `get`

### 文件: `factorminer/architecture/stages.py`

**模块说明**: Pluggable loop stages for Ralph and Helix iteration execution.

#### 类: `IterationPayload`

**说明**: Mutable state passed through each loop stage.

#### 类: `LoopStage`

**说明**: Abstract loop stage.

##### 方法: `LoopStage.run`


#### 类: `RetrieveStage`

**说明**: Retrieve memory priors for the next generation step.

##### 方法: `RetrieveStage.__init__`


##### 方法: `RetrieveStage.run`

- **内部/外部调用**: `_retrieve_fn`, `get_state_summary`

#### 类: `GenerateStage`

**说明**: Generate candidate formulas from prompt context.

##### 方法: `GenerateStage.__init__`


##### 方法: `GenerateStage.run`

- **内部/外部调用**: `_generate_fn`

#### 类: `EvaluateStage`

**说明**: Evaluate candidates under the active admission protocol.

##### 方法: `EvaluateStage.__init__`


##### 方法: `EvaluateStage.run`

- **内部/外部调用**: `_evaluate_fn`

#### 类: `LibraryUpdateStage`

**说明**: Apply admissions and replacements to the library.

##### 方法: `LibraryUpdateStage.__init__`


##### 方法: `LibraryUpdateStage.run`

- **内部/外部调用**: `_update_fn`

#### 类: `DistillStage`

**说明**: Distill evaluated trajectories back into memory.

##### 方法: `DistillStage.__init__`


##### 方法: `DistillStage.run`

- **内部/外部调用**: `_distill_fn`

### 文件: `factorminer/benchmark/__init__.py`

**模块说明**: Canonical benchmark runtime surface.

`factorminer.benchmark.runtime` is the primary benchmark API.
Legacy Helix benchmarking helpers are preserved behind lazy, deprecated
exports so older scripts keep working while the runtime path stays canonical.

#### 函数: `__getattr__`

- **内部/外部调用**: `AttributeError`, `warn`, `globals`, `import_module`

### 文件: `factorminer/benchmark/ablation.py`

**模块说明**: Runtime ablation study for HelixFactor Phase 2 components.

This module now drives ablations through the real loop path:
- HelixLoop execution on a training slice
- runtime recomputation of the admitted library
- freeze/top-k selection and combo evaluation on a held-out slice
- optional memory suppression via temporary monkeypatching

Supported ablations:
  full             - all components enabled
  no_debate        - disable specialist debate
  no_causal        - disable causal validation
  no_canonicalize  - disable SymPy deduplication
  no_regime        - disable regime-aware evaluation
  no_online_memory - disable memory retrieval / formation / evolution hooks
  no_capacity      - disable capacity estimation
  no_significance  - disable significance filtering
  no_memory        - disable memory-guided generation and updates

#### 类: `AblatedMethodRunner`

**说明**: Run one ablation variant through the real HelixLoop benchmark path.

##### 方法: `AblatedMethodRunner.__init__`

- **内部/外部调用**:

##### 方法: `AblatedMethodRunner._run_loop`

- **说明**: Instantiate and run the real HelixLoop on the training slice.
- **内部/外部调用**: `max`, `ExperienceMemory`, `TemporaryDirectory`, `_build_runtime_dataset`, `HelixLoop`, `MockProvider`, `_build_phase2_configs`, `min`, `run`, `FactorLibrary`, `_patched_memory_hooks`, `_build_mining_config`, `asarray`, `get`

##### 方法: `AblatedMethodRunner.run`

- **说明**: Run this ablation variant using the real loop + runtime contract.
- **内部/外部调用**: `_run_loop`, `_build_combined_dataset`, `_evaluate_runtime_library`, `_build_split_dataset`, `perf_counter`

#### 类: `AblationStudy`

**说明**: Run real-loop ablations and summarize component contribution.

##### 方法: `AblationStudy.__init__`


##### 方法: `AblationStudy.run_ablation`

- **说明**: Run one or more ablation variants on the real loop pipeline.
- **内部/外部调用**: `summarize_contributions`, `warning`, `_slice_data`, `MethodResult`, `keys`, `perf_counter`, `run`, `AblatedMethodRunner`, `info`, `AblationResult`, `get`

##### 方法: `AblationStudy.summarize_contributions`

- **说明**: Summarize component contributions relative to the full runtime run.
- **内部/外部调用**: `get`, `warning`, `max`, `reset_index`, `abs`, `append`, `sign`, `DataFrame`, `items`, `sort_values`

##### 方法: `AblationStudy.to_latex_table`

- **说明**: Generate a LaTeX ablation study table.
- **内部/外部调用**: `join`, `replace`, `iterrows`, `append`

##### 方法: `AblationStudy.print_summary`

- **说明**: Print a human-readable ablation summary.
- **内部/外部调用**: `replace`, `get`, `iterrows`

#### 函数: `_merge_slices`

- **说明**: Concatenate train/test slices into one runtime evaluation dictionary.
- **内部/外部调用**: `concatenate`, `sorted`, `asarray`

#### 函数: `_slice_data`

- **说明**: Slice all 2-D benchmark arrays to a column range.
- **内部/外部调用**: `items`

#### 函数: `_build_runtime_dataset`

- **说明**: Build a minimal runtime dataset from the benchmark dictionary format.
- **内部/外部调用**: `EvaluationDataset`, `stack`, `DatasetSplit`, `DataFrame`, `asarray`, `arange`

#### 函数: `_build_split_dataset`

- **说明**: Create a single-split runtime dataset from one benchmark slice.
- **内部/外部调用**: `_build_runtime_dataset`, `DatasetSplit`, `arange`

#### 函数: `_build_combined_dataset`

- **说明**: Create a train/test runtime dataset from sliced benchmark inputs.
- **内部/外部调用**: `_build_runtime_dataset`, `DatasetSplit`, `asarray`, `_merge_slices`, `arange`

#### 函数: `_build_mining_config`

- **说明**: Create a loop config tailored for a single runtime ablation.
- **内部/外部调用**: `MiningConfig`, `max`

#### 函数: `_build_phase2_configs`

- **说明**: Translate ablation flags into real HelixLoop runtime configs.
- **内部/外部调用**: `RuntimeDebateConfig`, `RuntimeCausalConfig`, `RuntimeCapacityConfig`, `RuntimeSignificanceConfig`, `RuntimeRegimeConfig`, `get`

#### 函数: `_patched_memory_hooks`

- **说明**: Disable memory retrieval and learning when a no-memory ablation is requested.
- **内部/外部调用**: `append`

#### 函数: `_compute_avg_abs_rho`

- **内部/外部调用**: `reshape`, `mean`, `corrcoef`, `isfinite`, `abs`, `triu_indices_from`

#### 函数: `_runtime_payload_to_result`

- **说明**: Convert runtime benchmark output into a MethodResult.
- **内部/外部调用**: `MethodResult`, `max`, `get`

#### 函数: `_evaluate_runtime_library`

- **说明**: Recompute a mined library using the runtime benchmark contract.
- **内部/外部调用**: `get`, `evaluate_factors`, `list_factors`, `evaluate_frozen_set`, `max`, `select_frozen_top_k`, `abs`, `build_benchmark_library`, `_runtime_payload_to_result`, `_compute_avg_abs_rho`

#### 函数: `run_full_ablation_study`

- **说明**: Run the full runtime ablation study on mock data.
- **内部/外部调用**: `values`, `AblationStudy`, `print_summary`, `MockProvider`, `keys`, `run_ablation`, `_build_mock_data_dict`

### 文件: `factorminer/benchmark/catalogs.py`

**模块说明**: Deterministic baseline formula catalogs for benchmark workflows.

#### 类: `CandidateEntry`

**说明**: One benchmark candidate formula.

#### 函数: `build_alpha101_adapted`

- **说明**: Expand the classic catalog into frequency-adapted window variants.
- **内部/外部调用**: `sub`, `CandidateEntry`, `append`

#### 函数: `build_random_exploration`

- **说明**: Generate deterministic random-formula candidates from safe templates.
- **内部/外部调用**: `choice`, `RandomState`, `CandidateEntry`, `format`, `append`, `range`, `randint`

#### 函数: `build_gplearn_style`

- **说明**: Build deeper deterministic mutation chains that mimic GP search.
- **内部/外部调用**: `max`, `RandomState`, `CandidateEntry`, `append`, `range`, `randint`, `build_random_exploration`

#### 函数: `build_alphaforge_style`

- **说明**: Reuse a diverse subset of the paper catalog for dynamic-combine baselines.
- **内部/外部调用**: `CandidateEntry`, `enumerate`, `append`

#### 函数: `build_alphaagent_style`

- **说明**: Reuse an alternate paper-catalog slice for LLM-style baseline proposals.
- **内部/外部调用**: `CandidateEntry`, `enumerate`, `append`

#### 函数: `build_factor_miner_catalog`

- **说明**: Expose the full paper factor catalog as benchmark candidates.
- **内部/外部调用**: `CandidateEntry`, `enumerate`

#### 函数: `entries_from_library`

- **说明**: Convert a saved FactorLibrary into benchmark candidate entries.
- **内部/外部调用**: `list_factors`, `CandidateEntry`

#### 函数: `dedupe_entries`

- **说明**: Remove duplicate formulas while preserving order.
- **内部/外部调用**: `add`, `append`

### 文件: `factorminer/benchmark/helix_benchmark.py`

**模块说明**: Legacy HelixBenchmark reports.

`factorminer.benchmark.runtime` is the canonical benchmark engine.
This module remains only for backward-compatible reports and CLI entry points.

CLI usage:
  python -m factorminer.benchmark.helix_benchmark --mock --n-factors 40 --output results/

#### 类: `MethodResult`

**说明**: Metrics for a single method run.

##### 方法: `MethodResult.to_dict`

- **内部/外部调用**: `pop`, `asdict`

#### 类: `DMTestResult`

**说明**: Diebold-Mariano test for forecast accuracy difference.

#### 类: `AblationResult`

**说明**: Result of one ablation study.

##### 方法: `AblationResult.to_dict`

- **内部/外部调用**: `to_dict`, `items`

#### 类: `OperatorSpeedResult`

**说明**: Timing for individual operators.

#### 类: `PipelineSpeedResult`

**说明**: Timing for end-to-end pipeline.

#### 类: `BenchmarkResult`

**说明**: Aggregate benchmark results — all methods, all metrics.

##### 方法: `BenchmarkResult.to_latex_table`

- **说明**: Generate a LaTeX table matching paper Table 1 style.
- **内部/外部调用**: `replace`, `isna`, `join`, `append`, `_g`, `fmt`

##### 方法: `BenchmarkResult.to_markdown_table`

- **说明**: Generate a Markdown table for GitHub README.
- **内部/外部调用**: `isna`, `join`, `append`, `_g`

##### 方法: `BenchmarkResult.plot_comparison`

- **说明**: Generate bar chart comparison (requires matplotlib).
- **内部/外部调用**: `set_title`, `set_xticks`, `mkdir`, `bar`, `subplots`, `append`, `set_visible`, `close`, `info`, `warning`, `tight_layout`, `suptitle`, `get`, `enumerate`, `replace`, `savefig`, `grid`, `range`, `set_xticklabels`, `Path`

##### 方法: `BenchmarkResult.generate_full_report`

- **说明**: Generate a complete HTML report with all results.
- **内部/外部调用**: `_build_html_report`, `open`, `mkdir`, `write`, `info`, `Path`

##### 方法: `BenchmarkResult._build_html_report`

- **内部/外部调用**: `to_html`, `join`, `strftime`, `append`, `now`, `items`

#### 类: `StatisticalComparisonTests`

**说明**: Rigorous statistical comparison between HelixFactor and FactorMiner.

##### 方法: `StatisticalComparisonTests.__init__`

- **内部/外部调用**: `RandomState`

##### 方法: `StatisticalComparisonTests._paired_valid_series`

- **说明**: Align paired series and drop rows with NaNs in either series.
- **内部/外部调用**: `min`, `isnan`, `asarray`

##### 方法: `StatisticalComparisonTests.diebold_mariano_test`

- **说明**: Diebold-Mariano test for forecast accuracy differences.
- **内部/外部调用**: `max`, `mean`, `DMTestResult`, `isfinite`, `abs`, `sqrt`, `var`, `cdf`, `allclose`, `range`, `_paired_valid_series`, `isnan`

##### 方法: `StatisticalComparisonTests.paired_t_test`

- **说明**: Paired t-test on IC difference series.
- **内部/外部调用**: `ttest_rel`, `mean`, `isfinite`, `_paired_valid_series`

##### 方法: `StatisticalComparisonTests.bootstrap_ic_difference_ci`

- **说明**: 95% block-bootstrap CI on mean IC difference.
- **内部/外部调用**: `max`, `ceil`, `mean`, `concatenate`, `percentile`, `min`, `range`, `randint`, `empty`, `_paired_valid_series`, `arange`

##### 方法: `StatisticalComparisonTests.wilcoxon_test`

- **说明**: Wilcoxon signed-rank test (non-parametric) on IC pairs.
- **内部/外部调用**: `_paired_valid_series`, `wilcoxon`

##### 方法: `StatisticalComparisonTests.run_all_tests`

- **说明**: Run all four statistical tests and return combined results.
- **内部/外部调用**: `paired_t_test`, `mean`, `wilcoxon_test`, `diebold_mariano_test`, `_paired_valid_series`, `bootstrap_ic_difference_ci`

#### 类: `SpeedBenchmark`

**说明**: Benchmark factor evaluation speed across operators and pipelines.

##### 方法: `SpeedBenchmark.__init__`

- **内部/外部调用**: `RandomState`

##### 方法: `SpeedBenchmark._time_callable`

- **说明**: Return minimum time over n_repeats (ms) after warmup runs.
- **内部/外部调用**: `fn`, `perf_counter`, `append`, `min`, `range`

##### 方法: `SpeedBenchmark.run_operator_benchmark`

- **说明**: Benchmark individual operators (numpy backend).
- **内部/外部调用**: `sqrt`, `std`, `RandomState`, `_ts_std`, `randn`, `_cs_rank`, `full_like`, `randint`, `sum`, `OperatorSpeedResult`, `isnan`, `items`, `_ts_rank`, `rankdata`, `_time_callable`, `mean`, `sliding_window_view`, `astype`, `range`, `_ts_corr`

##### 方法: `SpeedBenchmark.run_full_pipeline_benchmark`

- **说明**: Benchmark end-to-end candidate evaluation pipeline.
- **内部/外部调用**: `values`, `compute_ic`, `max`, `evaluate`, `PipelineSpeedResult`, `compute_ic_mean`, `try_parse`, `perf_counter`, `randn`, `build_random_exploration`, `get`, `_build_mock_data_dict`

##### 方法: `SpeedBenchmark.generate_speed_table`

- **说明**: Generate a LaTeX table of speed results.
- **内部/外部调用**: `values`, `max`, `join`, `append`, `items`

#### 类: `HelixBenchmark`

**说明**: Rigorous comparison of HelixFactor vs FactorMiner (and baselines).

##### 方法: `HelixBenchmark.__init__`

- **内部/外部调用**: `_warn_legacy_runtime`, `SpeedBenchmark`, `StatisticalComparisonTests`

##### 方法: `HelixBenchmark.run_comparison`

- **说明**: Run the full comparison benchmark.
- **内部/外部调用**: `run_operator_benchmark`, `_slice_data`, `BenchmarkResult`, `_average_method_results`, `_build_selection_df`, `_build_combination_df`, `append`, `info`, `warning`, `_build_library_df`, `run_full_pipeline_benchmark`, `run_all_tests`, `items`, `get`, `MethodResult`, `_synthetic_ic_series`, `run_single_method`, `range`, `_build_speed_df`

##### 方法: `HelixBenchmark.run_single_method`

- **说明**: Run one method and return its MethodResult.
- **内部/外部调用**: `_evaluate_candidates`, `_build_library`, `max`, `_library_metrics`, `warning`, `MethodResult`, `_get_candidates`, `perf_counter`, `_selection_metrics`, `_combination_metrics`, `get`

##### 方法: `HelixBenchmark._get_candidates`

- **说明**: Get candidate (name, formula, category) tuples for a method.
- **内部/外部调用**: `spec_from_file_location`, `max`, `module_from_spec`, `build_alpha101_adapted`, `exec_module`, `build_factor_miner_catalog`, `build_random_exploration`, `Path`

##### 方法: `HelixBenchmark._evaluate_candidates`

- **说明**: Evaluate candidates; returns list of result dicts.
- **内部/外部调用**: `compute_ic`, `evaluate`, `compute_ic_mean`, `try_parse`, `append`, `compute_ic_win_rate`, `compute_icir`, `all`, `isnan`

##### 方法: `HelixBenchmark._build_library`

- **说明**: Build a diversified factor library with IC and correlation admission.
- **内部/外部调用**: `compute_pairwise_correlation`, `abs`, `append`, `sort`, `get`

##### 方法: `HelixBenchmark._library_metrics`

- **说明**: Compute library IC, ICIR, avg|rho|. Returns (ic, icir, rho, ic_series).
- **内部/外部调用**: `mean`, `nanmean`, `abs`, `compute_pairwise_correlation`, `stack`, `append`, `min`, `range`, `get`

##### 方法: `HelixBenchmark._combination_metrics`

- **说明**: Compute EW/ICW combination metrics on test data.
- **内部/外部调用**: `compute_ic`, `FactorCombiner`, `compute_ic_mean`, `ic_weighted`, `compute_icir`, `get`, `enumerate`, `equal_weight`

##### 方法: `HelixBenchmark._selection_metrics`

- **说明**: Compute Lasso/XGBoost selection IC on test data.
- **内部/外部调用**: `_evaluate_candidates`, `compute_ic`, `xgboost_selection`, `debug`, `nanmean`, `stack`, `compute_ic_mean`, `FactorSelector`, `compute_icir`, `lasso_selection`, `get`, `enumerate`

##### 方法: `HelixBenchmark._clone_cfg`

- **内部/外部调用**: `deepcopy`

##### 方法: `HelixBenchmark._build_runtime_provider`

- **内部/外部调用**: `warning`, `MockProvider`, `create_provider`, `get`

##### 方法: `HelixBenchmark._build_runtime_mining_config`

- **内部/外部调用**: `RuntimeMiningConfig`

##### 方法: `HelixBenchmark._build_debate_config`

- **内部/外部调用**: `RuntimeDebateConfig`, `min`

##### 方法: `HelixBenchmark._runtime_phase2_kwargs`

- **内部/外部调用**: `target_cls`, `values`, `_build_debate_config`, `get`, `_clone_section`

##### 方法: `HelixBenchmark._execute_runtime_loop`

- **内部/外部调用**: `select_frozen_top_k`, `abs`, `HelixLoop`, `stack`, `exists`, `min`, `mkdir`, `run`, `load`, `RalphLoop`, `evaluate_frozen_set`, `load_library`, `evaluate_factors`, `_build_runtime_provider`, `list_factors`, `sorted`, `open`, `update`, `resolve`, `get`, `items`, `mean`, `MethodResult`, `nanmean`, `get_summary`, `_runtime_phase2_kwargs`, `_build_runtime_mining_config`

##### 方法: `HelixBenchmark._runtime_method_frames`

- **内部/外部调用**: `get`, `append`, `DataFrame`, `items`

##### 方法: `HelixBenchmark.run_runtime_comparison`

- **说明**: Run a benchmark with real Ralph/Helix executions for Phase 2.
- **内部/外部调用**: `run_operator_benchmark`, `_slice_by_indices`, `mkdir`, `BenchmarkResult`, `_average_method_results`, `_build_selection_df`, `load_benchmark_dataset`, `_build_combination_df`, `append`, `_build_library_df`, `run_full_pipeline_benchmark`, `resolve`, `run_all_tests`, `_runtime_method_frames`, `items`, `get`, `setdefault`, `_synthetic_ic_series`, `_execute_runtime_loop`, `run_single_method`, `range`, `_build_speed_df`

##### 方法: `HelixBenchmark.run_runtime_ablation_study`

- **说明**: Run a runtime-backed ablation study using real loop executions.
- **内部/外部调用**: `warning`, `append`, `_execute_runtime_loop`, `items`, `mkdir`, `DataFrame`, `_clone_cfg`, `AblationResult`, `get`, `load_benchmark_dataset`

#### 函数: `_warn_legacy_runtime`

- **内部/外部调用**: `warn`

#### 函数: `_json_safe`

- **说明**: Recursively convert a structure into JSON-safe primitives.
- **内部/外部调用**: `_json_safe`, `isfinite`, `items`, `item`

#### 函数: `_build_mock_data_dict`

- **说明**: Build a minimal data dict from MockConfig (no raw_df needed).
- **内部/外部调用**: `generate_mock_data`, `preprocess`, `sorted`, `pivot`, `astype`, `min`, `MockConfig`, `unique`, `groupby`, `size`, `items`, `roll`

#### 函数: `_slice_data`

- **说明**: Slice all (M, T) arrays to columns [start, end).
- **内部/外部调用**: `items`

#### 函数: `_average_method_results`

- **说明**: Average numeric fields across multiple runs.
- **内部/外部调用**: `mean`, `MethodResult`

#### 函数: `_build_library_df`

- **内部/外部调用**: `MethodResult`, `append`, `get`, `DataFrame`

#### 函数: `_build_combination_df`

- **内部/外部调用**: `MethodResult`, `append`, `get`, `DataFrame`

#### 函数: `_build_selection_df`

- **内部/外部调用**: `max`, `MethodResult`, `append`, `DataFrame`, `get`

#### 函数: `_build_speed_df`

- **内部/外部调用**: `DataFrame`, `items`, `append`

#### 函数: `_synthetic_ic_series`

- **说明**: Generate a synthetic IC series with given mean for stat tests.
- **内部/外部调用**: `astype`, `randn`, `RandomState`

#### 函数: `_parse_args`

- **内部/外部调用**: `ArgumentParser`, `add_argument`, `parse_args`

#### 函数: `main`

- **内部/外部调用**: `dump`, `mkdir`, `_build_mock_data_dict`, `values`, `_json_safe`, `plot_comparison`, `basicConfig`, `generate_full_report`, `_warn_legacy_runtime`, `_parse_args`, `debug`, `open`, `to_markdown_table`, `upper`, `write`, `run_comparison`, `get`, `to_csv`, `to_string`, `perf_counter`, `HelixBenchmark`, `to_latex_table`, `Path`

### 文件: `factorminer/benchmark/runtime.py`

**模块说明**: Strict paper/research benchmark runners built on runtime recomputation.

#### 类: `BenchmarkManifest`

**说明**: Serializable description of one benchmark run.

#### 类: `WalkForwardBenchmarkContract`

**说明**: Canonical train/test freeze contract used by the benchmark runtime.

##### 方法: `WalkForwardBenchmarkContract.to_dict`

- **内部/外部调用**: `_json_safe`, `asdict`

#### 类: `StressBenchmarkContract`

**说明**: Canonical transaction-cost and capacity stress contract.

##### 方法: `StressBenchmarkContract.to_dict`

- **内部/外部调用**: `_json_safe`, `asdict`

#### 类: `StrategyGridBenchmarkContract`

**说明**: Canonical strategy grid for benchmark ablations.

##### 方法: `StrategyGridBenchmarkContract.to_dict`

- **内部/外部调用**: `_json_safe`, `asdict`

#### 类: `BenchmarkRuntimeContract`

**说明**: Complete runtime benchmark contract emitted to manifests/results.

##### 方法: `BenchmarkRuntimeContract.to_dict`

- **内部/外部调用**: `to_dict`, `_json_safe`

#### 函数: `_clone_cfg`

- **内部/外部调用**: `deepcopy`

#### 函数: `_cfg_with_overrides`

- **内部/外部调用**: `_clone_cfg`

#### 函数: `_data_hash`

- **内部/外部调用**: `tobytes`, `hexdigest`, `reset_index`, `sha256`, `update`, `hash_pandas_object`, `sort_values`

#### 函数: `_json_safe`

- **说明**: Recursively convert NaN/inf values into JSON-safe nulls.
- **内部/外部调用**: `_json_safe`, `isnan`, `items`, `isinf`, `item`

#### 函数: `_file_sha256`

- **内部/外部调用**: `read`, `hexdigest`, `sha256`, `open`, `iter`, `update`

#### 函数: `_json_summary`

- **内部/外部调用**: `type`, `open`, `exists`, `load`

#### 函数: `_session_summary`

- **内部/外部调用**: `load`, `get_summary`, `exists`

#### 函数: `_catalog_provenance`


#### 函数: `_saved_library_provenance`

- **内部/外部调用**: `get_diagnostics`, `_json_summary`, `with_suffix`, `_file_sha256`, `items`, `resolve`, `exists`, `expanduser`, `_base_path`, `_session_summary`, `Path`, `load_library`

#### 函数: `_baseline_provenance`

- **内部/外部调用**: `_saved_library_provenance`, `_catalog_provenance`

#### 函数: `_runtime_manifest_value`

- **说明**: Return the runtime manifest for one baseline if supplied.
- **内部/外部调用**: `get`

#### 函数: `_default_capacity_levels`

- **内部/外部调用**: `RuntimeCapacityConfig`

#### 函数: `_safe_len`

- **内部/外部调用**:

#### 函数: `_build_walk_forward_contract`

- **说明**: Build the canonical walk-forward benchmark contract.
- **内部/外部调用**: `WalkForwardBenchmarkContract`

#### 函数: `_build_stress_contract`

- **说明**: Build the canonical cost/capacity stress contract.
- **内部/外部调用**: `RuntimeCapacityConfig`, `StressBenchmarkContract`, `_default_capacity_levels`, `get`

#### 函数: `_build_strategy_grid_contract`

- **说明**: Build the canonical memory-policy / dependence / backend strategy grid.
- **内部/外部调用**: `_strategy_ablation_raw_config`, `StrategyGridBenchmarkContract`, `get`

#### 函数: `_benchmark_dataset_contract`

- **说明**: Build a safe, benchmark-local summary of the frozen dataset.
- **内部/外部调用**: `max`, `_safe_len`, `keys`, `shape`, `items`

#### 函数: `build_benchmark_runtime_contract`

- **说明**: Build the canonical benchmark runtime contract for one baseline.
- **内部/外部调用**: `_build_walk_forward_contract`, `BenchmarkRuntimeContract`, `from_config`, `runtime_contract`, `_build_stress_contract`, `_build_strategy_grid_contract`

#### 函数: `_build_runtime_provider`

- **说明**: Create the benchmark-time LLM provider.
- **内部/外部调用**: `MockProvider`, `get`, `create_provider`

#### 函数: `_filter_dataclass_kwargs`

- **说明**: Copy shared dataclass fields from one config object to another.
- **内部/外部调用**: `fields`

#### 函数: `_build_phase2_runtime_kwargs`

- **说明**: Build runtime Phase 2 configs from the hierarchical benchmark config.
- **内部/外部调用**: `_filter_dataclass_kwargs`, `RuntimeDebateConfig`, `RuntimeCapacityConfig`, `RuntimeCausalConfig`, `RuntimeSignificanceConfig`, `RuntimeRegimeConfig`

#### 函数: `_extract_volume_panel`

- **说明**: Best-effort extraction of a dollar-volume panel for Helix capacity checks.
- **内部/外部调用**: `any`, `get`, `isfinite`, `asarray`

#### 函数: `_split_volume_panel`

- **说明**: Align the available volume panel to one dataset split.
- **内部/外部调用**: `get_split`, `_extract_volume_panel`, `asarray`

#### 函数: `_capacity_pressure_summary`

- **说明**: Compute a compact capacity-stress summary for one factor/composite.
- **内部/外部调用**: `CapacityEstimator`, `RuntimeCapacityConfig`, `estimate`, `asarray`, `items`

#### 函数: `_build_runtime_loop_config`

- **说明**: Build the flat loop config consumed by RalphLoop/HelixLoop.
- **内部/外部调用**: `max`, `items`, `min`, `LoopMiningConfig`, `get`

#### 函数: `_cfg_for_runtime_baseline`

- **说明**: Project the hierarchical config into one runtime benchmark variant.
- **内部/外部调用**: `_clone_cfg`

#### 函数: `_real_mining_loop_type`

- **说明**: Resolve the loop type for a runtime mining request.
- **内部/外部调用**: `lower`, `strip`, `get`

#### 函数: `_runtime_loop_provenance`

- **说明**: Summarize the real mining run used to source benchmark factors.
- **内部/外部调用**: `get_diagnostics`, `_json_safe`, `_json_summary`, `_file_sha256`, `items`, `exists`, `_session_summary`, `load_library`

#### 函数: `_run_runtime_mining_loop`

- **说明**: Run a real RalphLoop/HelixLoop and return its factor library.
- **内部/外部调用**: `RalphLoop`, `_build_runtime_provider`, `_cfg_for_runtime_baseline`, `load_session`, `HelixLoop`, `_build_runtime_loop_config`, `_real_mining_loop_type`, `_extract_volume_panel`, `run`, `_ensure_dir`, `_build_phase2_runtime_kwargs`, `get`, `_runtime_loop_provenance`

#### 函数: `load_benchmark_dataset`

- **说明**: Load one universe into the canonical runtime dataset.
- **内部/外部调用**: `get`, `generate_mock_data`, `lower`, `_data_hash`, `_cfg_with_overrides`, `load_runtime_dataset`, `MockConfig`, `load_market_data`

#### 函数: `_factors_from_entries`

- **内部/外部调用**: `Factor`, `enumerate`

#### 函数: `_get_baseline_entries`

- **内部/外部调用**: `dedupe_entries`, `build_alphaagent_style`, `KeyError`, `entries_from_library`, `build_alpha101_adapted`, `build_gplearn_style`, `build_alphaforge_style`, `_base_path`, `build_factor_miner_catalog`, `build_random_exploration`, `load_library`

#### 函数: `_base_path`

- **内部/外部调用**: `with_suffix`, `Path`

#### 函数: `build_benchmark_library`

- **说明**: Build a library from candidate artifacts under the paper admission rules.
- **内部/外部调用**: `admit_factor`, `check_replacement`, `replace_factor`, `abs`, `_max_correlation_with_library`, `FactorLibrary`, `check_admission`, `Factor`, `sort`

#### 函数: `select_frozen_top_k`

- **说明**: Freeze the paper Top-K set from train-split recomputed metrics.
- **内部/外部调用**: `list_factors`, `extend`, `abs`, `sort`

#### 函数: `_abs_icir_from_series`

- **内部/外部调用**: `mean`, `isfinite`, `abs`, `std`

#### 函数: `_normalize_backtest_stats`

- **内部/外部调用**: `abs`, `_abs_icir_from_series`, `asarray`, `get`

#### 函数: `_avg_abs_rho`

- **内部/外部调用**: `mean`, `abs`, `compute_correlation_matrix`, `triu_indices_from`

#### 函数: `_weighted_composite`

- **内部/外部调用**: `zeros_like`, `abs`, `sum`, `get`

#### 函数: `evaluate_frozen_set`

- **说明**: Evaluate one frozen factor set on one universe.
- **内部/外部调用**: `_split_volume_panel`, `PortfolioBacktester`, `FactorCombiner`, `abs`, `orthogonal`, `_factors_from_entries`, `tolist`, `forward_stepwise`, `_json_safe`, `xgboost_selection`, `append`, `lasso_selection`, `evaluate_factors`, `get_split`, `CandidateEntry`, `sign`, `ic_weighted`, `quintile_backtest`, `items`, `get`, `equal_weight`, `mean`, `_normalize_backtest_stats`, `FactorSelector`, `_default_capacity_levels`, `asarray`, `_weighted_composite`, `_capacity_pressure_summary`, `_avg_abs_rho`

#### 函数: `_ensure_dir`

- **内部/外部调用**: `mkdir`

#### 函数: `_write_json`

- **内部/外部调用**: `dump`, `mkdir`, `open`, `_json_safe`

#### 函数: `_save_manifest`

- **内部/外部调用**: `_write_json`, `asdict`

#### 函数: `_strategy_ablation_raw_config`

- **内部/外部调用**: `get`

#### 函数: `_runtime_strategy_backends`

- **内部/外部调用**: `lower`, `torch_available`, `append`, `strip`, `c_backend_available`, `get`

#### 函数: `_mean_universe_metric`

- **内部/外部调用**: `values`, `mean`, `append`, `get`

#### 函数: `run_table1_benchmark`

- **说明**: Run the strict Top-K freeze benchmark across all configured universes.
- **内部/外部调用**: `_write_json`, `select_frozen_top_k`, `abs`, `_cfg_with_overrides`, `_factors_from_entries`, `_ensure_dir`, `load_benchmark_dataset`, `_runtime_manifest_value`, `evaluate_frozen_set`, `to_dict`, `evaluate_factors`, `list_factors`, `_save_manifest`, `build_benchmark_library`, `BenchmarkManifest`, `get`, `_benchmark_dataset_contract`, `_run_runtime_mining_loop`, `build_benchmark_runtime_contract`, `_get_baseline_entries`, `_baseline_provenance`

#### 函数: `run_ablation_memory_benchmark`

- **说明**: Compare the default FactorMiner lane to the relaxed no-memory lane.
- **内部/外部调用**: `get`, `_write_json`, `max`, `run_table1_benchmark`, `_ensure_dir`, `items`

#### 函数: `run_ablation_strategy_benchmark`

- **说明**: Compare runtime loop variants across memory policy × dependence × backend.
- **内部/外部调用**: `_write_json`, `_runtime_strategy_backends`, `_cfg_with_overrides`, `_ensure_dir`, `load_benchmark_dataset`, `_runtime_manifest_value`, `to_dict`, `append`, `_strategy_ablation_raw_config`, `_build_strategy_grid_contract`, `max`, `sorted`, `get`, `_benchmark_dataset_contract`, `build_benchmark_runtime_contract`, `run_table1_benchmark`, `_mean_universe_metric`

#### 函数: `run_cost_pressure_benchmark`

- **说明**: Run cost-pressure analysis for one baseline on the configured universes.
- **内部/外部调用**: `get`, `_write_json`, `run_table1_benchmark`, `_ensure_dir`, `items`

#### 函数: `_time_callable`

- **内部/外部调用**: `fn`, `append`, `perf_counter`, `min`, `range`

#### 函数: `run_efficiency_benchmark`

- **说明**: Benchmark operator-level and factor-level compute time.
- **内部/外部调用**: `_write_json`, `_backend_inputs`, `_time_callable`, `to_tensor`, `RandomState`, `plot_efficiency_benchmark`, `torch_available`, `execute_operator`, `astype`, `randn`, `full_like`, `c_backend_available`, `_ensure_dir`, `r`, `items`

#### 函数: `run_benchmark_suite`

- **说明**: Run the benchmark suite and return the artifact index.
- **内部/外部调用**: `get`, `_write_json`, `run_ablation_strategy_benchmark`, `_strategy_ablation_raw_config`, `run_ablation_memory_benchmark`, `run_table1_benchmark`, `_ensure_dir`, `run_cost_pressure_benchmark`, `run_efficiency_benchmark`

#### 函数: `run_runtime_mining_benchmark`

- **说明**: Run the benchmark suite with explicit real-loop manifests when provided.
- **内部/外部调用**: `run_benchmark_suite`

### 文件: `factorminer/cli.py`

**模块说明**: Click-based CLI for FactorMiner.

#### 函数: `_setup_logging`

- **说明**: Configure root logger for CLI output.
- **内部/外部调用**: `basicConfig`

#### 函数: `_load_data`

- **说明**: Load market data from file or generate mock data.
- **内部/外部调用**: `Abort`, `generate_mock_data`, `echo`, `load_market_data`, `MockConfig`, `get`

#### 函数: `_prepare_data_arrays`

- **说明**: Convert a market DataFrame to numpy arrays for the mining loop.
- **内部/外部调用**: `unique`, `iterrows`, `where`, `sorted`, `abs`, `divide`, `full`, `full_like`, `range`, `index`, `isnan`, `enumerate`

#### 函数: `_create_llm_provider`

- **说明**: Create an LLM provider from config or use mock.
- **内部/外部调用**: `Abort`, `MockProvider`, `echo`, `create_provider`, `get`

#### 函数: `_build_core_mining_config`

- **说明**: Create the flat mining config expected by RalphLoop/HelixLoop.
- **内部/外部调用**: `CoreMiningConfig`

#### 函数: `_attach_runtime_targets`

- **说明**: Attach multi-horizon runtime metadata for research-mode mining.
- **内部/外部调用**: `max`, `items`

#### 函数: `_save_result_library`

- **说明**: Persist a factor library to the standard output location.
- **内部/外部调用**: `save_library`, `mkdir`, `with_suffix`

#### 函数: `_filter_dataclass_kwargs`

- **说明**: Copy shared dataclass fields from one config object to another.
- **内部/外部调用**: `fields`

#### 函数: `_build_debate_config`

- **说明**: Build the runtime debate config from YAML config settings.
- **内部/外部调用**: `warning`, `RuntimeDebateConfig`

#### 函数: `_build_phase2_runtime_configs`

- **说明**: Instantiate evaluation/runtime configs for the Helix loop.
- **内部/外部调用**: `_filter_dataclass_kwargs`, `RuntimeCausalConfig`, `RuntimeCapacityConfig`, `RuntimeSignificanceConfig`, `_build_debate_config`, `RuntimeRegimeConfig`

#### 函数: `_extract_capacity_volume`

- **说明**: Prefer dollar volume (`amount`) and fall back to raw volume if needed.
- **内部/外部调用**: `all`, `isnan`

#### 函数: `_active_phase2_features`

- **说明**: Describe the effective Helix feature set for CLI output.
- **内部/外部调用**: `append`

#### 函数: `_load_runtime_dataset_for_analysis`

- **说明**: Load, preprocess, split, and tensorize data for analysis commands.
- **内部/外部调用**: `load_runtime_dataset`, `_load_data`

#### 函数: `_recompute_analysis_artifacts`

- **说明**: Recompute library factors on the canonical analysis dataset.
- **内部/外部调用**: `evaluate_factors`, `list_factors`

#### 函数: `_report_artifact_failures`

- **说明**: Print a concise recomputation failure summary and return failure texts.
- **内部/外部调用**: `echo`, `summarize_failures`

#### 函数: `_artifact_map_by_id`


#### 函数: `_select_artifacts_for_ids`

- **内部/外部调用**: `Abort`, `_artifact_map_by_id`, `join`, `append`, `echo`, `get`

#### 函数: `_analysis_output_path`

- **内部/外部调用**:

#### 函数: `_print_benchmark_summary`

- **说明**: Emit a concise benchmark summary for CLI runs.
- **内部/外部调用**: `dumps`, `values`, `get`, `echo`, `all`, `items`

#### 函数: `_print_recomputed_factor_table`

- **内部/外部调用**: `echo`

#### 函数: `_print_split_summary`

- **内部/外部调用**: `max`, `mean`, `echo`, `min`

#### 函数: `_load_library_from_path`

- **说明**: Load a factor library, handling both .json extension and base path.
- **内部/外部调用**: `Abort`, `with_suffix`, `echo`, `Path`, `load_library`

#### 函数: `main`

- **说明**: FactorMiner -- LLM-powered quantitative factor mining.
- **内部/外部调用**: `Abort`, `version_option`, `load_config`, `open`, `safe_load`, `_setup_logging`, `setdefault`, `update`, `option`, `exists`, `echo`, `ensure_object`, `group`, `get`, `Path`

#### 函数: `validate_data`

- **说明**: Validate a market-data file before mining.
- **内部/外部调用**: `Abort`, `argument`, `validate_market_data`, `command`, `to_dict`, `Path`, `option`, `echo`, `exit`, `dumps`, `exit_code`, `render_validation_report`

#### 函数: `report`

- **说明**: Generate a static report from FactorMiner artifacts.
- **内部/外部调用**: `argument`, `generate_report`, `Choice`, `command`, `option`, `echo`, `Path`

#### 函数: `mine`

- **说明**: Run a factor mining session.
- **内部/外部调用**: `_build_core_mining_config`, `RalphLoop`, `_attach_runtime_targets`, `Abort`, `validate`, `command`, `exception`, `option`, `echo`, `run`, `_load_runtime_dataset_for_analysis`, `_save_result_library`, `_create_llm_provider`, `_load_library_from_path`, `get`, `Path`

#### 函数: `evaluate`

- **说明**: Evaluate a factor library on historical data.
- **内部/外部调用**: `Abort`, `argument`, `_recompute_analysis_artifacts`, `_print_recomputed_factor_table`, `command`, `Choice`, `analysis_split_names`, `Path`, `option`, `echo`, `_load_runtime_dataset_for_analysis`, `_report_artifact_failures`, `_load_library_from_path`, `select_top_k`, `_print_split_summary`

#### 函数: `combine`

- **说明**: Run factor combination and selection methods.
- **内部/外部调用**: `Abort`, `PortfolioBacktester`, `exception`, `FactorCombiner`, `orthogonal`, `_load_library_from_path`, `run_research_model_suite`, `select_top_k`, `forward_stepwise`, `argument`, `xgboost_selection`, `resolve_split_for_fit_eval`, `option`, `_load_runtime_dataset_for_analysis`, `capitalize`, `lasso_selection`, `_artifact_map_by_id`, `get_split`, `upper`, `ic_weighted`, `quintile_backtest`, `_report_artifact_failures`, `items`, `dumps`, `equal_weight`, `get`, `_recompute_analysis_artifacts`, `Choice`, `command`, `FactorSelector`, `echo`, `write_text`, `Path`

#### 函数: `visualize`

- **说明**: Generate plots and tear sheets for a factor library.
- **内部/外部调用**: `Abort`, `mkdir`, `plot_quintile_returns`, `FactorTearSheet`, `plot_correlation_heatmap`, `_load_library_from_path`, `select_top_k`, `argument`, `_select_artifacts_for_ids`, `compute_correlation_matrix`, `option`, `_load_runtime_dataset_for_analysis`, `plot_ic_timeseries`, `get_split`, `analysis_split_names`, `generate`, `_report_artifact_failures`, `_recompute_analysis_artifacts`, `_analysis_output_path`, `command`, `Choice`, `echo`, `range`, `Path`

#### 函数: `export_cmd`

- **说明**: Export a factor library to various formats.
- **内部/外部调用**: `Abort`, `argument`, `export_formulas`, `Choice`, `command`, `exception`, `with_suffix`, `option`, `save_library`, `echo`, `mkdir`, `export_csv`, `_load_library_from_path`, `Path`

#### 函数: `benchmark`

- **说明**: Run strict paper/research benchmark workflows.
- **内部/外部调用**: `group`

#### 函数: `_benchmark_common_options`

- **内部/外部调用**: `option`, `pass_context`, `Path`

#### 函数: `benchmark_table1`

- **说明**: Run the Top-K freeze benchmark across configured universes.
- **内部/外部调用**: `command`, `option`, `_print_benchmark_summary`, `run_table1_benchmark`

#### 函数: `benchmark_ablation_memory`

- **说明**: Run the experience-memory ablation benchmark.
- **内部/外部调用**: `_print_benchmark_summary`, `command`, `run_ablation_memory_benchmark`

#### 函数: `benchmark_ablation_strategy`

- **说明**: Run runtime ablations across memory policy, dependence metric, and backend.
- **内部/外部调用**: `_print_benchmark_summary`, `option`, `run_ablation_strategy_benchmark`, `command`

#### 函数: `benchmark_cost_pressure`

- **说明**: Run transaction-cost pressure testing.
- **内部/外部调用**: `_print_benchmark_summary`, `option`, `run_cost_pressure_benchmark`, `command`

#### 函数: `benchmark_efficiency`

- **说明**: Run operator-level and factor-level efficiency benchmarks.
- **内部/外部调用**: `_print_benchmark_summary`, `run_efficiency_benchmark`, `command`

#### 函数: `benchmark_suite`

- **说明**: Run the full benchmark suite.
- **内部/外部调用**: `_print_benchmark_summary`, `command`, `run_benchmark_suite`

#### 函数: `helix`

- **说明**: Run the enhanced Helix Loop with Phase 2 features.
- **内部/外部调用**: `Abort`, `join`, `exception`, `HelixLoop`, `run`, `_load_library_from_path`, `_extract_capacity_volume`, `option`, `_load_runtime_dataset_for_analysis`, `_build_phase2_runtime_configs`, `_attach_runtime_targets`, `_save_result_library`, `_create_llm_provider`, `get`, `_build_core_mining_config`, `validate`, `command`, `echo`, `_active_phase2_features`, `Path`

### 文件: `factorminer/configs/__init__.py`

**模块说明**: Configuration defaults and schemas for FactorMiner.

#### 函数: `load_default_yaml`

- **说明**: Load the default YAML configuration shipped with the package.
- **内部/外部调用**: `exists`, `open`, `safe_load`

### 文件: `factorminer/core/__init__.py`

**模块说明**: FactorMiner core: expression trees, types, factor DSL parser, and Ralph Loop.

#### 函数: `__getattr__`

- **内部/外部调用**: `AttributeError`, `import_module`, `globals`

### 文件: `factorminer/core/canonicalizer.py`

**模块说明**: SymPy-based formula canonicalization for duplicate detection.

Converts ``ExpressionTree`` objects into canonical SymPy expressions so that
algebraically equivalent formulas (e.g. ``Add($close, $open)`` vs
``Add($open, $close)``, or ``Neg(Neg($close))`` vs ``$close``) produce
identical hashes.

**Design principle**: Arithmetic operators map to native SymPy math so that
standard simplifications (commutativity, double-negation, x/x = 1, etc.) are
applied automatically.  Non-algebraic operators (rolling windows,
cross-sectional transforms, conditionals) are represented as opaque
``sympy.Function`` symbols so their structure is preserved without false
simplification.

#### 类: `FormulaCanonicalizer`

**说明**: Canonicalize expression trees via SymPy simplification.

##### 方法: `FormulaCanonicalizer.__init__`


##### 方法: `FormulaCanonicalizer.canonicalize`

- **说明**: Return an MD5 hash of the canonical (simplified) form of *tree*.
- **内部/外部调用**: `hexdigest`, `to_string`, `encode`, `md5`, `_tree_to_sympy`, `simplify`

##### 方法: `FormulaCanonicalizer.is_duplicate`

- **说明**: Return ``True`` if *tree_a* and *tree_b* are algebraically equivalent.
- **内部/外部调用**: `canonicalize`

##### 方法: `FormulaCanonicalizer.get_canonical_form`

- **说明**: Return the simplified string representation (not hashed).
- **内部/外部调用**: `_tree_to_sympy`, `simplify`

##### 方法: `FormulaCanonicalizer.clear_cache`

- **说明**: Discard all cached canonical hashes.
- **内部/外部调用**: `clear`

##### 方法: `FormulaCanonicalizer._tree_to_sympy`

- **说明**: Recursively convert an expression-tree node to a SymPy expression.
- **内部/外部调用**: `Float`, `type`, `_map_operator`, `_tree_to_sympy`, `Symbol`

##### 方法: `FormulaCanonicalizer._map_operator`

- **说明**: Dispatch an operator to its SymPy equivalent.
- **内部/外部调用**: `Float`, `sorted`, `sqrt`, `append`, `log`, `func`, `Function`, `Abs`

### 文件: `factorminer/core/config.py`

**模块说明**: Mining-specific configuration for the Ralph Loop.

Provides a flat configuration dataclass specifically for the mining loop,
separate from the hierarchical Config system in utils/config.py.  This
allows the RalphLoop to accept a simple, focused parameter object while
the full Config handles loading, validation, and serialization.

#### 类: `MiningConfig`

**说明**: Flat configuration controlling the Ralph Loop mining process.

##### 方法: `MiningConfig.validate`

- **说明**: Basic sanity checks on parameter values.
- **内部/外部调用**:

### 文件: `factorminer/core/expression_tree.py`

**模块说明**: Expression tree data structure for alpha-factor formulas.

An expression tree is a DAG of ``Node`` objects whose leaves are raw
market-data features (``LeafNode``) or numeric constants (``ConstantNode``)
and whose internal nodes are operator applications (``OperatorNode``).

#### 类: `Node`

**说明**: Abstract base for every node in an expression tree.

##### 方法: `Node.evaluate`

- **说明**: Compute the node's value given market data.

##### 方法: `Node.to_string`

- **说明**: Serialize the subtree rooted at this node to a DSL formula.

##### 方法: `Node.depth`

- **说明**: Return the depth of the subtree (leaf = 1).

##### 方法: `Node.size`

- **说明**: Return the number of nodes in the subtree.

##### 方法: `Node.clone`

- **说明**: Return a deep copy of the subtree.

##### 方法: `Node.__repr__`

- **内部/外部调用**: `to_string`

##### 方法: `Node.iter_nodes`

- **说明**: Yield every node in the subtree (pre-order).
- **内部/外部调用**: `iter_nodes`

##### 方法: `Node.leaf_features`

- **说明**: Return sorted unique feature names referenced by this subtree.
- **内部/外部调用**: `sorted`, `iter_nodes`

#### 类: `LeafNode`

**说明**: References a raw market-data column (e.g. ``$close``).

##### 方法: `LeafNode.__init__`

- **内部/外部调用**: `sorted`

##### 方法: `LeafNode.evaluate`

- **内部/外部调用**: `astype`, `sorted`, `keys`, `KeyError`

##### 方法: `LeafNode.to_string`


##### 方法: `LeafNode.depth`


##### 方法: `LeafNode.size`


##### 方法: `LeafNode.clone`

- **内部/外部调用**: `LeafNode`

#### 类: `ConstantNode`

**说明**: A numeric literal embedded in the expression.

##### 方法: `ConstantNode.__init__`

- **内部/外部调用**:

##### 方法: `ConstantNode.evaluate`

- **内部/外部调用**: `values`, `full_like`

##### 方法: `ConstantNode.to_string`

- **内部/外部调用**: `abs`

##### 方法: `ConstantNode.depth`


##### 方法: `ConstantNode.size`


##### 方法: `ConstantNode.clone`

- **内部/外部调用**: `ConstantNode`

#### 类: `OperatorNode`

**说明**: An internal node that applies an operator to child sub-trees.

##### 方法: `OperatorNode.__init__`

- **内部/外部调用**: `items`

##### 方法: `OperatorNode.to_string`

- **内部/外部调用**: `to_string`, `join`, `abs`, `append`

##### 方法: `OperatorNode.depth`

- **内部/外部调用**: `max`, `depth`

##### 方法: `OperatorNode.size`

- **内部/外部调用**: `sum`, `size`

##### 方法: `OperatorNode.clone`

- **内部/外部调用**: `OperatorNode`, `clone`

##### 方法: `OperatorNode.evaluate`

- **内部/外部调用**: `_dispatch_operator`, `evaluate`

#### 类: `ExpressionTree`

**说明**: Wrapper around a root ``Node`` providing a convenient API.

##### 方法: `ExpressionTree.__init__`


##### 方法: `ExpressionTree.to_string`

- **说明**: Serialize the full tree to a DSL formula string.
- **内部/外部调用**: `to_string`

##### 方法: `ExpressionTree.depth`

- **说明**: Return the depth of the tree.
- **内部/外部调用**: `depth`

##### 方法: `ExpressionTree.size`

- **说明**: Return the total number of nodes.
- **内部/外部调用**: `size`

##### 方法: `ExpressionTree.evaluate`

- **说明**: Execute the formula on market data.
- **内部/外部调用**: `evaluate`

##### 方法: `ExpressionTree.clone`

- **说明**: Return a deep copy of the tree.
- **内部/外部调用**: `ExpressionTree`, `clone`

##### 方法: `ExpressionTree.leaf_features`

- **说明**: Return sorted unique feature names referenced by this tree.
- **内部/外部调用**: `leaf_features`

##### 方法: `ExpressionTree.__repr__`

- **内部/外部调用**: `to_string`

##### 方法: `ExpressionTree.__str__`

- **内部/外部调用**: `to_string`

#### 函数: `_safe_div`

- **说明**: Division that returns 0 where the denominator is near zero.
- **内部/外部调用**: `where`, `abs`

#### 函数: `_safe_log`

- **内部/外部调用**: `sign`, `abs`, `log1p`

#### 函数: `_safe_sqrt`

- **内部/外部调用**: `sign`, `abs`, `sqrt`

#### 函数: `_rolling_apply`

- **说明**: Apply *func* over a rolling window along the last axis (T).
- **内部/外部调用**: `func`, `full_like`, `range`

#### 函数: `_ts_mean`

- **内部/外部调用**: `nanmean`

#### 函数: `_ts_std`

- **内部/外部调用**: `nanstd`

#### 函数: `_ts_var`

- **内部/外部调用**: `nanvar`

#### 函数: `_ts_sum`

- **内部/外部调用**: `nansum`

#### 函数: `_ts_prod`

- **内部/外部调用**: `nanprod`

#### 函数: `_ts_max`

- **内部/外部调用**: `nanmax`

#### 函数: `_ts_min`

- **内部/外部调用**: `nanmin`

#### 函数: `_ts_argmax`

- **内部/外部调用**: `astype`, `nanargmax`

#### 函数: `_ts_argmin`

- **内部/外部调用**: `astype`, `nanargmin`

#### 函数: `_ts_median`

- **内部/外部调用**: `nanmedian`

#### 函数: `_ts_skew`

- **内部/外部调用**: `nanstd`, `where`, `nanmean`, `max`

#### 函数: `_ts_kurt`

- **内部/外部调用**: `nanstd`, `where`, `nanmean`

#### 函数: `_ts_rank`

- **说明**: Percentile rank of the latest value within the window.
- **内部/外部调用**: `sum`, `astype`

#### 函数: `_ts_corr`

- **内部/外部调用**: `nanstd`, `where`, `nanmean`

#### 函数: `_ts_cov`

- **内部/外部调用**: `nanmean`

#### 函数: `_ts_beta`

- **说明**: Rolling OLS slope of x on y.
- **内部/外部调用**: `nansum`, `where`, `nanmean`

#### 函数: `_ts_resid`

- **内部/外部调用**: `_ts_beta`, `nanmean`, `squeeze`

#### 函数: `_ema`

- **说明**: Exponential moving average along the last axis.
- **内部/外部调用**: `empty_like`, `range`

#### 函数: `_wma`

- **说明**: Linearly-weighted moving average.
- **内部/外部调用**: `sum`, `arange`, `full_like`, `range`

#### 函数: `_decay`

- **说明**: Exponentially decaying sum.
- **内部/外部调用**: `sum`, `array`, `full_like`, `range`

#### 函数: `_cs_rank`

- **说明**: Cross-sectional percentile rank at each time step.
- **内部/外部调用**: `sum`, `astype`, `any`, `empty_like`, `range`, `empty`, `argsort`, `isnan`

#### 函数: `_cs_zscore`

- **内部/外部调用**: `max`, `nanmean`, `nanstd`, `empty_like`, `range`

#### 函数: `_cs_demean`

- **内部/外部调用**: `nanmean`

#### 函数: `_cs_scale`

- **内部/外部调用**: `nansum`, `where`, `abs`

#### 函数: `_ts_linreg_slope`

- **内部/外部调用**: `nansum`, `max`, `mean`, `nanmean`, `full_like`, `range`, `sum`, `arange`

#### 函数: `_ts_linreg_intercept`

- **内部/外部调用**: `mean`, `nanmean`, `full_like`, `range`, `_ts_linreg_slope`, `arange`

#### 函数: `_ts_linreg_fitted`

- **内部/外部调用**: `_ts_linreg_intercept`, `_ts_linreg_slope`

#### 函数: `_ts_linreg_resid`

- **内部/外部调用**: `_ts_linreg_fitted`, `full_like`, `range`

#### 函数: `_dispatch_operator`

- **说明**: Execute an operator on evaluated children, return result array.
- **内部/外部调用**: `maximum`, `_rolling_apply`, `_cs_scale`, `abs`, `sqrt`, `minimum`, `nan_to_num`, `nancumprod`, `_decay`, `_ts_linreg_resid`, `log`, `_safe_div`, `_cs_rank`, `max`, `where`, `sign`, `_safe_log`, `clip`, `nancumsum`, `full_like`, `accumulate`, `_wma`, `sum`, `get`, `_cs_zscore`, `_safe_sqrt`, `isnan`, `_ema`, `ceil`, `_cs_demean`, `_ts_linreg_fitted`, `ones_like`, `astype`, `nanquantile`, `_ts_linreg_intercept`, `NotImplementedError`, `_ts_linreg_slope`

### 文件: `factorminer/core/factor_library.py`

**模块说明**: Factor Library: maintains the growing collection of admitted alpha factors.

Implements the admission rules from the paper (Eq. 10, 11):
- Admission: IC(alpha) >= tau_IC AND max_{g in L} |rho(alpha, g)| < theta
- Replacement: IC(alpha) >= 0.10 AND IC(alpha) >= 1.3 * IC(g) AND only 1 correlated factor

The library tracks pairwise Spearman correlations and supports incremental
updates as new factors are added or replaced.

#### 类: `Factor`

**说明**: A single admitted alpha factor.

##### 方法: `Factor.__post_init__`

- **内部/外部调用**: `now`, `strftime`

##### 方法: `Factor.to_dict`

- **说明**: Serialize to a JSON-compatible dictionary (excludes signals).

##### 方法: `Factor.from_dict`

- **说明**: Reconstruct a Factor from a dictionary.
- **内部/外部调用**: `cls`, `get`

#### 类: `FactorLibrary`

**说明**: The factor library L that maintains admitted alpha factors.

##### 方法: `FactorLibrary.__init__`

- **内部/外部调用**: `build_dependence_metric`

##### 方法: `FactorLibrary.compute_correlation`

- **说明**: Compute time-average dependence rho(alpha, beta) under the active metric.
- **内部/外部调用**: `compute`

##### 方法: `FactorLibrary._compute_correlation_vectorized`

- **说明**: Compatibility alias for the active dependence metric implementation.
- **内部/外部调用**: `compute_correlation`

##### 方法: `FactorLibrary.check_admission`

- **说明**: Check if candidate passes admission criteria (Eq. 10).
- **内部/外部调用**: `_max_correlation_with_library`

##### 方法: `FactorLibrary.check_replacement`

- **说明**: Check replacement mechanism (Eq. 11).
- **内部/外部调用**: `append`, `_compute_correlation_vectorized`, `items`

##### 方法: `FactorLibrary.admit_factor`

- **说明**: Add a factor to the library and update the correlation matrix.
- **内部/外部调用**: `info`, `_extend_correlation_matrix`

##### 方法: `FactorLibrary.replace_factor`

- **说明**: Replace an existing factor with a better one.
- **内部/外部调用**: `pop`, `info`, `_recompute_matrix_slot`, `KeyError`

##### 方法: `FactorLibrary.remove_factor`

- **说明**: Remove a factor from the library and rebuild correlation state.
- **内部/外部调用**: `pop`, `info`, `update_correlation_matrix`, `KeyError`

##### 方法: `FactorLibrary._max_correlation_with_library`

- **说明**: Compute max |rho| between candidate and all library factors.
- **内部/外部调用**: `values`, `_compute_correlation_vectorized`, `max`

##### 方法: `FactorLibrary._extend_correlation_matrix`

- **说明**: Extend the correlation matrix by one row/column for the new factor.
- **内部/外部调用**: `zeros`, `items`, `_compute_correlation_vectorized`, `range`, `get`

##### 方法: `FactorLibrary._recompute_matrix_slot`

- **说明**: Recompute one row/column of the correlation matrix after replacement.
- **内部/外部调用**: `items`, `_compute_correlation_vectorized`, `get`, `range`

##### 方法: `FactorLibrary.update_correlation_matrix`

- **说明**: Recompute the full pairwise correlation matrix from scratch.
- **内部/外部调用**: `zeros`, `sorted`, `keys`, `_compute_correlation_vectorized`, `range`, `clear`, `enumerate`

##### 方法: `FactorLibrary.size`

- **说明**: Number of factors currently in the library.
- **内部/外部调用**:

##### 方法: `FactorLibrary.get_factor`

- **说明**: Retrieve a factor by ID.
- **内部/外部调用**: `KeyError`

##### 方法: `FactorLibrary.list_factors`

- **说明**: Return all factors sorted by ID.
- **内部/外部调用**: `sorted`

##### 方法: `FactorLibrary.get_factors_by_category`

- **说明**: Return all factors matching a given category.
- **内部/外部调用**: `values`

##### 方法: `FactorLibrary.get_diagnostics`

- **说明**: Library diagnostics: avg |rho|, max tail correlations, per-category counts, saturation.
- **内部/外部调用**: `values`, `max`, `defaultdict`, `mean`, `percentile`, `isnan`, `triu_indices`

##### 方法: `FactorLibrary.get_state_summary`

- **说明**: Summary for memory retrieval: size, categories, recent admissions.
- **内部/外部调用**: `values`, `sorted`, `defaultdict`

### 文件: `factorminer/core/helix_loop.py`

**模块说明**: The Helix Loop: 5-stage self-evolving factor discovery with Phase 2 extensions.

Extends the base Ralph Loop with:
  1. RETRIEVE  -- KG + embeddings + flat memory hybrid retrieval
  2. PROPOSE   -- Multi-agent debate (specialists + critic) or standard generation
  3. SYNTHESIZE -- SymPy canonicalization to eliminate mathematical duplicates
  4. VALIDATE  -- Standard pipeline + causal + regime + capacity + significance
  5. DISTILL   -- Standard memory evolution + KG update + online forgetting

All Phase 2 components are optional: when none are enabled the Helix Loop
behaves identically to the Ralph Loop and is a full drop-in replacement.

#### 类: `HelixLoop`

**说明**: Enhanced 5-stage Helix Loop for self-evolving factor discovery.

##### 方法: `HelixLoop.__init__`

- **内部/外部调用**: `__init__`, `LoopExecutionService`, `_init_phase2_components`, `OnlineForgettingService`, `RetrieveStage`, `LibraryUpdateStage`, `update`, `GenerateStage`, `KnowledgeGraphService`, `EvaluateStage`, `DistillStage`

##### 方法: `HelixLoop._init_phase2_components`

- **说明**: Initialize all Phase 2 components based on configuration.
- **内部/外部调用**: `RegimeEvalCls`, `DebateGeneratorCls`, `_try_import_kg`, `RegimeDetectorCls`, `FormulaCanonCls`, `CustomStoreCls`, `_prime_embedder_from_library`, `_try_import_regime`, `_try_import_significance`, `info`, `warning`, `InventorCls`, `_try_import_causal`, `CapacityEstCls`, `KGCls`, `_try_import_capacity`, `_try_import_debate`, `classify`, `_try_import_custom_store`, `_try_import_canonicalizer`, `BootstrapCls`, `FDRCls`, `EmbedderCls`, `_try_import_auto_inventor`, `_try_import_embedder`, `Path`

##### 方法: `HelixLoop._run_iteration`

- **说明**: Execute one iteration of the 5-stage Helix Loop.
- **内部/外部调用**: `get_diagnostics`, `warning`, `empty_stats`, `record_compute`, `run_stage_chain`, `log_telemetry`, `describe_empty_generation`, `record_llm_call`, `new_payload`, `candidate_count`, `_phase2_features`, `time`, `build_telemetry`, `_attach_factor_provenance`, `get`, `_generator_family`, `build_stats`

##### 方法: `HelixLoop._stage_retrieve_helix`

- **内部/外部调用**: `_helix_retrieve`

##### 方法: `HelixLoop._stage_generate_helix`

- **内部/外部调用**: `_helix_propose`, `build`, `to_dict`, `_canonicalize_and_dedup`

##### 方法: `HelixLoop._stage_evaluate_helix`

- **内部/外部调用**: `record_batch_results`, `evaluate_batch`

##### 方法: `HelixLoop._stage_library_update_helix`

- **内部/外部调用**: `_helix_validate`, `_update_library`

##### 方法: `HelixLoop._stage_distill_helix`

- **内部/外部调用**: `_run_auto_invention`, `_stage_distill`, `_helix_distill`

##### 方法: `HelixLoop._helix_retrieve`

- **说明**: Stage 1 RETRIEVE: KG + embeddings + flat memory hybrid retrieval.
- **内部/外部调用**: `retrieve_memory`, `_try_import_kg_retrieval`, `retrieve_enhanced_fn`, `warning`

##### 方法: `HelixLoop._helix_propose`

- **说明**: Stage 2 PROPOSE: Use debate generator or standard generator.
- **内部/外部调用**: `warning`, `append`, `generate_batch`, `info`

##### 方法: `HelixLoop._canonicalize_and_dedup`

- **说明**: Stage 3 SYNTHESIZE: Remove mathematically equivalent candidates.
- **内部/外部调用**: `debug`, `append`, `try_parse`, `_semantic_duplicate_target`, `canonicalize`, `info`

##### 方法: `HelixLoop._helix_validate`

- **说明**: Stage 4 extended VALIDATE: causal + regime + capacity + significance.
- **内部/外部调用**: `_validate_significance`, `_validate_capacity`, `any`, `_validate_causal`, `info`, `_validate_regime`

##### 方法: `HelixLoop._validate_causal`

- **说明**: Run causal validation (Granger + intervention) on admitted candidates.
- **内部/外部调用**: `warning`, `list_factors`, `validate`, `_revoke_admission`, `debug`, `CausalValidatorCls`, `_try_import_causal`

##### 方法: `HelixLoop._validate_regime`

- **说明**: Run regime-aware IC evaluation on admitted candidates.
- **内部/外部调用**: `warning`, `_revoke_admission`, `evaluate`, `debug`

##### 方法: `HelixLoop._validate_capacity`

- **说明**: Run capacity-aware cost evaluation on admitted candidates.
- **内部/外部调用**: `warning`, `_revoke_admission`, `debug`, `net_cost_evaluation`

##### 方法: `HelixLoop._validate_significance`

- **说明**: Run bootstrap CI + batch-level FDR correction on admitted candidates.
- **内部/外部调用**: `get`, `compute_ic`, `warning`, `_revoke_admission`, `debug`, `batch_evaluate`, `items`

##### 方法: `HelixLoop._revoke_admission`

- **说明**: Revoke a previously admitted candidate from the library.
- **内部/外部调用**: `warning`, `list_factors`, `_remove_semantic_artifacts`, `debug`, `remove_factor`

##### 方法: `HelixLoop._helix_distill`

- **说明**: Stage 5 DISTILL: KG update + embeddings + online forgetting.
- **内部/外部调用**: `_update_knowledge_graph`, `apply`, `debug`, `embed`

##### 方法: `HelixLoop._update_knowledge_graph`

- **说明**: Update the knowledge graph with new factor nodes and edges.
- **内部/外部调用**: `debug`, `add_result`

##### 方法: `HelixLoop._remove_semantic_artifacts`

- **说明**: Remove a factor from derived semantic stores if present.
- **内部/外部调用**: `remove_factor`

##### 方法: `HelixLoop._run_auto_invention`

- **说明**: Periodically propose, validate, and register new operators.
- **内部/外部调用**: `warning`, `propose_operators`, `validate_operator`, `_register_invented_operator`, `debug`, `record_llm_call`, `append`, `info`

##### 方法: `HelixLoop._register_invented_operator`

- **说明**: Register a validated auto-invented operator.
- **内部/外部调用**: `warning`, `register`, `OperatorSpec`, `CustomOperator`, `_compile_operator_code`, `info`, `items`

##### 方法: `HelixLoop._checkpoint`

- **说明**: Save a periodic checkpoint including Phase 2 state.
- **内部/外部调用**: `warning`, `save_session`

##### 方法: `HelixLoop.save_session`

- **说明**: Save the full mining session state including Phase 2 components.
- **内部/外部调用**: `dump`, `warning`, `save`, `debug`, `open`, `save_session`, `_persist_run_manifest`, `_refresh_run_manifest`, `Path`

##### 方法: `HelixLoop.load_session`

- **说明**: Resume a mining session from a saved checkpoint.
- **内部/外部调用**: `warning`, `get_factor_count`, `_try_import_kg`, `open`, `_prime_embedder_from_library`, `get_edge_count`, `exists`, `info`, `load_session`, `get`, `Path`, `load`

##### 方法: `HelixLoop._loop_type`

- **说明**: Label the loop for provenance and manifests.

##### 方法: `HelixLoop._phase2_features`

- **说明**: List the enabled Helix Phase 2 features.
- **内部/外部调用**: `append`

##### 方法: `HelixLoop._generator_family`

- **说明**: Return the active Helix generator label for provenance.
- **内部/外部调用**: `_generator_family`

##### 方法: `HelixLoop._extract_operators`

- **说明**: Extract operator names from a DSL formula string.
- **内部/外部调用**: `findall`

##### 方法: `HelixLoop._extract_features`

- **说明**: Extract feature names (e.g. $close, $volume) from a formula.
- **内部/外部调用**: `findall`

##### 方法: `HelixLoop._prime_embedder_from_library`

- **说明**: Seed the embedder cache from the currently admitted library.
- **内部/外部调用**: `debug`, `list_factors`, `clear`, `embed`

##### 方法: `HelixLoop._semantic_duplicate_target`

- **说明**: Return the matched library factor if embeddings flag a near-duplicate.
- **内部/外部调用**: `debug`, `is_semantic_duplicate`

#### 函数: `_try_import_debate`


#### 函数: `_try_import_canonicalizer`


#### 函数: `_try_import_causal`


#### 函数: `_try_import_regime`


#### 函数: `_try_import_capacity`


#### 函数: `_try_import_significance`


#### 函数: `_try_import_kg`


#### 函数: `_try_import_kg_retrieval`


#### 函数: `_try_import_embedder`


#### 函数: `_try_import_auto_inventor`


#### 函数: `_try_import_custom_store`


### 文件: `factorminer/core/library_io.py`

**模块说明**: Serialization and I/O for the FactorLibrary.

Provides save/load to JSON + optional binary signal cache (.npz),
CSV export, formula export, and import of the 110 factors from the paper.

#### 函数: `save_library`

- **说明**: Save a FactorLibrary to disk.
- **内部/外部调用**: `dump`, `tolist`, `info`, `list_factors`, `open`, `with_suffix`, `to_dict`, `savez_compressed`, `mkdir`, `items`, `Path`

#### 函数: `load_library`

- **说明**: Load a FactorLibrary from disk.
- **内部/外部调用**: `values`, `info`, `open`, `with_suffix`, `from_dict`, `items`, `exists`, `close`, `FactorLibrary`, `array`, `get`, `Path`, `load`

#### 函数: `export_csv`

- **说明**: Export the factor table to CSV.
- **内部/外部调用**: `list_factors`, `writerow`, `open`, `DictWriter`, `writeheader`, `mkdir`, `info`, `Path`

#### 函数: `export_formulas`

- **说明**: Export just the formulas for reproduction.
- **内部/外部调用**: `list_factors`, `open`, `mkdir`, `write`, `info`, `Path`

#### 函数: `import_from_paper`

- **说明**: Import the 110 factors from the paper's Appendix P.
- **内部/外部调用**: `info`, `admit_factor`, `open`, `Path`, `FactorLibrary`, `Factor`, `get`, `enumerate`, `load`

### 文件: `factorminer/core/loop_services.py`

**模块说明**: Reusable orchestration helpers for Ralph and Helix loops.

This module keeps the loop classes focused on policy decisions while
centralizing the repeated stage-chain execution and iteration telemetry.

#### 类: `IterationTelemetry`

**说明**: Structured snapshot of one completed iteration.

#### 类: `LoopExecutionService`

**说明**: Shared orchestration helpers for factor-mining loops.

##### 方法: `LoopExecutionService.__init__`


##### 方法: `LoopExecutionService.new_payload`

- **内部/外部调用**: `IterationPayload`

##### 方法: `LoopExecutionService.run_stage_chain`

- **内部/外部调用**: `run`

##### 方法: `LoopExecutionService.candidate_count`

- **内部/外部调用**:

##### 方法: `LoopExecutionService.empty_stats`

- **内部/外部调用**: `_empty_stats`

##### 方法: `LoopExecutionService.build_stats`

- **内部/外部调用**: `_compute_stats`, `update`

##### 方法: `LoopExecutionService.describe_empty_generation`

- **内部/外部调用**: `candidate_count`

##### 方法: `LoopExecutionService.zero_admission_guidance`

- **说明**: Explain likely causes when a run produced candidates but admitted nothing.
- **内部/外部调用**: `get_summary`, `get`

##### 方法: `LoopExecutionService.build_telemetry`

- **内部/外部调用**: `candidate_count`, `IterationTelemetry`

##### 方法: `LoopExecutionService.log_telemetry`

- **说明**: Emit the shared iteration telemetry to reporter and session logs.
- **内部/外部调用**: `IterationRecord`, `max`, `mean`, `log_factor`, `log_iteration`, `log_batch`, `get`, `FactorRecord`

### 文件: `factorminer/core/parser.py`

**模块说明**: Recursive-descent parser for the FactorMiner factor DSL.

Converts string formulas such as::

    Neg(CsRank(Div(Sub($close, $vwap), $vwap)))

into ``ExpressionTree`` objects backed by the operator registry defined in
:mod:`factorminer.core.types`.

Grammar (informal)
------------------

::

    expression  := function_call | feature_ref | number
    function_call := IDENTIFIER '(' arg_list ')'
    arg_list    := expression (',' expression)*
    feature_ref := '$' IDENTIFIER
    number      := ['-'] DIGITS ['.' DIGITS] [('e'|'E') ['-'|'+'] DIGITS]

Usage
-----

>>> from factorminer.core.parser import parse
>>> tree = parse("Neg(Div(Sub($close, $vwap), $vwap))")
>>> tree.to_string()
'Neg(Div(Sub($close, $vwap), $vwap))'

#### 类: `TokenType`

#### 类: `Token`

##### 方法: `Token.__repr__`


#### 类: `Parser`

**说明**: Recursive-descent parser that converts a token stream to a ``Node``.

##### 方法: `Parser.__init__`


##### 方法: `Parser._peek`


##### 方法: `Parser._advance`


##### 方法: `Parser._expect`

- **内部/外部调用**: `SyntaxError`, `_advance`

##### 方法: `Parser.parse_expression`

- **说明**: Parse a single expression (the start symbol).
- **内部/外部调用**: `SyntaxError`, `_peek`, `_parse_number`, `_parse_function_call`, `_parse_feature`

##### 方法: `Parser._parse_feature`

- **内部/外部调用**: `SyntaxError`, `sorted`, `LeafNode`, `_advance`

##### 方法: `Parser._parse_number`

- **内部/外部调用**: `SyntaxError`, `_advance`, `ConstantNode`

##### 方法: `Parser._parse_function_call`

- **说明**: Parse ``Name(arg1, arg2, ..., paramN)``.
- **内部/外部调用**: `OperatorNode`, `SyntaxError`, `sorted`, `_advance`, `keys`, `append`, `_parse_arg`, `_peek`, `_expect`, `get`, `enumerate`

##### 方法: `Parser._parse_arg`

- **说明**: Parse a single argument inside a function call.
- **内部/外部调用**: `_advance`, `_peek`, `parse_expression`, `ConstantNode`

#### 函数: `tokenize`

- **说明**: Convert a formula string into a list of ``Token`` objects.
- **内部/外部调用**: `match`, `Token`, `SyntaxError`, `append`, `end`, `isdigit`, `isalpha`, `group`

#### 函数: `parse`

- **说明**: Parse a factor formula string into an ``ExpressionTree``.
- **内部/外部调用**: `SyntaxError`, `strip`, `Parser`, `tokenize`, `parse_expression`, `_peek`, `ExpressionTree`

#### 函数: `try_parse`

- **说明**: Like :func:`parse` but returns ``None`` on failure instead of raising.
- **内部/外部调用**: `parse`

### 文件: `factorminer/core/provenance.py`

**模块说明**: Run and factor provenance helpers for mining sessions.

This module keeps provenance data compact, JSON-safe, and stable across
save/load boundaries.

#### 类: `RunManifest`

**说明**: Serializable description of a mining run.

##### 方法: `RunManifest.to_dict`

- **内部/外部调用**: `_json_safe`, `asdict`

#### 类: `FactorProvenance`

**说明**: Serializable provenance payload attached to an admitted factor.

##### 方法: `FactorProvenance.to_dict`

- **内部/外部调用**: `_json_safe`, `asdict`

#### 函数: `_json_safe`

- **说明**: Recursively convert common scientific Python objects into JSON-safe data.
- **内部/外部调用**: `_json_safe`, `asdict`, `tolist`, `is_dataclass`, `items`, `item`

#### 函数: `stable_digest`

- **说明**: Compute a stable SHA256 digest for a JSON-serializable payload.
- **内部/外部调用**: `_json_safe`, `hexdigest`, `encode`, `sha256`, `dumps`

#### 函数: `_compact_reference_list`

- **说明**: Normalize a mixed list of factor references into readable strings.
- **内部/外部调用**: `add`, `append`, `strip`, `get`

#### 函数: `_compact_memory_signal`

- **说明**: Keep only the most useful pieces of memory context.
- **内部/外部调用**: `get`, `_json_safe`, `_compact_reference_list`

#### 函数: `build_run_manifest`

- **说明**: Build a run manifest from the live loop state.
- **内部/外部调用**: `_json_safe`, `stable_digest`, `RunManifest`

#### 函数: `build_factor_provenance`

- **说明**: Build per-factor provenance from the current mining context.
- **内部/外部调用**: `_compact_memory_signal`, `isoformat`, `_json_safe`, `FactorProvenance`, `now`, `get`

### 文件: `factorminer/core/ralph_loop.py`

**模块说明**: The Ralph Loop: self-evolving factor discovery algorithm.

Implements Algorithm 1 from the FactorMiner paper.  The loop iteratively:
  1. Retrieves memory priors from experience memory  -- R(M, L)
  2. Generates candidate factors via LLM guided by memory -- G(m, L)
  3. Evaluates candidates through a multi-stage pipeline:
     - Stage 1: Fast IC screening on M_fast assets
     - Stage 2: Correlation check against library L
     - Stage 2.5: Replacement check for correlated candidates
     - Stage 3: Intra-batch deduplication (pairwise rho < theta)
     - Stage 4: Full validation on M_full assets + trajectory collection
  4. Updates the factor library with admitted factors  -- L <- L + {alpha}
  5. Evolves the experience memory with new insights   -- E(M, F(M, tau))

The loop terminates when the library reaches the target size K or the
maximum number of iterations is exhausted.

#### 类: `BudgetTracker`

**说明**: Tracks resource consumption across the mining session.

##### 方法: `BudgetTracker.record_llm_call`


##### 方法: `BudgetTracker.record_compute`


##### 方法: `BudgetTracker.wall_elapsed`

- **内部/外部调用**: `time`

##### 方法: `BudgetTracker.total_tokens`


##### 方法: `BudgetTracker.is_exhausted`

- **说明**: True if any budget limit has been reached.

##### 方法: `BudgetTracker.to_dict`

- **内部/外部调用**: `round`

#### 类: `EvaluationResult`

**说明**: Result of evaluating a single candidate factor.

#### 类: `FactorGenerator`

**说明**: Generates candidate factors using LLM guided by memory priors.

##### 方法: `FactorGenerator.__init__`

- **内部/外部调用**: `MockProvider`, `PromptBuilder`

##### 方法: `FactorGenerator.generate_batch`

- **说明**: Generate a batch of candidate factors.
- **内部/外部调用**: `_parse_response`, `generate`, `build_user_prompt`

##### 方法: `FactorGenerator._parse_response`

- **说明**: Parse LLM output into (name, formula) pairs.
- **内部/外部调用**: `match`, `append`, `strip`, `splitlines`, `group`

#### 类: `ValidationPipeline`

**说明**: Multi-stage evaluation pipeline for candidate factors.

##### 方法: `ValidationPipeline.__init__`

- **内部/外部调用**: `choice`, `type`, `RandomState`, `EvaluationKernel`, `LibraryGeometry`, `from_config`, `FactorLibrary`, `sort`, `arange`

##### 方法: `ValidationPipeline.evaluate_candidate`

- **说明**: Evaluate a single candidate through the full pipeline.
- **内部/外部调用**: `_research_replacement`, `EvaluationResult`, `compute_quality_score`, `compute_factor_stats`, `replacement_decision`, `list_factors`, `admission_decision`, `_research_enabled`, `compute_signals`, `candidate_geometry`, `_build_data_dict`, `compute_target_stats`, `all`, `isnan`

##### 方法: `ValidationPipeline._research_enabled`

- **内部/外部调用**:

##### 方法: `ValidationPipeline._research_replacement`

- **内部/外部调用**: `list_factors`, `max`, `get_factor`, `compute_correlation`, `append`, `get`

##### 方法: `ValidationPipeline.evaluate_batch`

- **说明**: Evaluate a batch through all stages including intra-batch dedup.
- **内部/外部调用**: `evaluate_candidate`, `_deduplicate_batch`, `_evaluate_parallel`, `append`

##### 方法: `ValidationPipeline._evaluate_parallel`

- **说明**: Evaluate candidates using a thread pool.
- **内部/外部调用**: `ThreadPoolExecutor`, `evaluate_candidate`, `submit`, `as_completed`, `result`, `enumerate`

##### 方法: `ValidationPipeline._deduplicate_batch`

- **说明**: Stage 3: Remove intra-batch duplicates among admitted candidates.
- **内部/外部调用**: `debug`, `sum`, `deduplicate_results`, `_research_enabled`

##### 方法: `ValidationPipeline._build_data_dict`

- **说明**: Convert data_tensor to a dict mapping feature names to (M, T) arrays.
- **内部/外部调用**: `enumerate`

##### 方法: `ValidationPipeline._compute_signals`

- **说明**: Compute factor signals from expression tree on the data tensor.
- **内部/外部调用**: `_build_data_dict`, `compute_tree_signals`

#### 类: `MiningReporter`

**说明**: Lightweight reporter that logs batch results to a JSONL file.

##### 方法: `MiningReporter.__init__`

- **内部/外部调用**: `mkdir`, `Path`

##### 方法: `MiningReporter.log_batch`

- **说明**: Append a batch record to the JSONL log.
- **内部/外部调用**: `open`, `update`, `time`, `write`, `dumps`

##### 方法: `MiningReporter.export_library`

- **说明**: Export the factor library to JSON.
- **内部/外部调用**: `get_diagnostics`, `isoformat`, `dump`, `list_factors`, `open`, `to_dict`, `now`

#### 类: `RalphLoop`

**说明**: Self-Evolving Factor Discovery via the Ralph Loop paradigm.

##### 方法: `RalphLoop.__init__`

- **说明**: Initialize the Ralph Loop.
- **内部/外部调用**: `LoopExecutionService`, `EvaluationKernel`, `FactorGenerator`, `MiningReporter`, `from_arrays`, `PromptBuilder`, `EvaluateStage`, `FactorFamilyDiscovery`, `BudgetTracker`, `ExperienceMemory`, `LibraryGeometry`, `RetrieveStage`, `GenerateStage`, `FactorLifecycleStore`, `build_memory_policy`, `DistillStage`, `LibraryUpdateStage`, `from_config`, `PromptContextBuilder`, `FactorLibrary`, `ValidationPipeline`, `FactorAdmissionService`

##### 方法: `RalphLoop.run`

- **说明**: Run the complete mining loop.
- **内部/外部调用**: `schema`, `exists`, `log_session_end`, `_persist_run_manifest`, `_serialize_config`, `export_library`, `save`, `MiningSession`, `finalize`, `to_dict`, `log_session_start`, `start_progress`, `info`, `BudgetTracker`, `_checkpoint`, `warning`, `callback`, `time`, `runtime_contract`, `now`, `_refresh_run_manifest`, `record_iteration`, `get`, `_run_iteration`, `MiningSessionLogger`, `strftime`, `is_exhausted`, `zero_admission_guidance`, `load_session`, `Path`

##### 方法: `RalphLoop._run_iteration`

- **说明**: Execute one iteration of the Ralph Loop.
- **内部/外部调用**: `get_diagnostics`, `warning`, `empty_stats`, `record_compute`, `run_stage_chain`, `log_telemetry`, `record_llm_call`, `new_payload`, `time`, `build_telemetry`, `_attach_factor_provenance`, `_generator_family`, `build_stats`

##### 方法: `RalphLoop._stage_retrieve`

- **内部/外部调用**: `retrieve`

##### 方法: `RalphLoop._stage_generate`

- **内部/外部调用**: `generate_batch`, `to_dict`, `build`

##### 方法: `RalphLoop._stage_evaluate`

- **内部/外部调用**: `record_batch_results`, `evaluate_batch`

##### 方法: `RalphLoop._stage_library_update`

- **内部/外部调用**: `_update_library`

##### 方法: `RalphLoop._stage_distill`

- **内部/外部调用**: `record_memory_distillation`, `form`, `_build_trajectory`, `evolve`

##### 方法: `RalphLoop._update_library`

- **说明**: Admit passing factors into the library and handle replacements.
- **内部/外部调用**: `admit_results`

##### 方法: `RalphLoop._build_trajectory`

- **说明**: Build mining trajectory tau for memory formation.
- **内部/外部调用**: `build_trajectory`, `append`

##### 方法: `RalphLoop._compute_stats`

- **说明**: Compute per-iteration statistics.
- **内部/外部调用**: `get_diagnostics`, `max`, `lower`, `to_dict`, `sum`, `get`

##### 方法: `RalphLoop._empty_stats`

- **说明**: Return empty stats dict for iterations with no candidates.
- **内部/外部调用**: `to_dict`

##### 方法: `RalphLoop._infer_category`

- **说明**: Infer factor category from formula structure.
- **内部/外部调用**: `infer_family`

##### 方法: `RalphLoop.save_session`

- **说明**: Save the full mining session state for resume.
- **内部/外部调用**: `serialize`, `dump`, `info`, `save`, `open`, `save_library`, `mkdir`, `_persist_run_manifest`, `_refresh_run_manifest`, `Path`, `startswith`

##### 方法: `RalphLoop.load_session`

- **说明**: Resume a mining session from a saved checkpoint.
- **内部/外部调用**: `info`, `load`, `open`, `exists`, `restore`, `get`, `Path`, `load_library`

##### 方法: `RalphLoop.resume_from`

- **说明**: Create a RalphLoop and restore state from a checkpoint.
- **内部/外部调用**: `cls`, `load_session`

##### 方法: `RalphLoop._checkpoint`

- **说明**: Save a periodic checkpoint.
- **内部/外部调用**: `warning`, `save_session`

##### 方法: `RalphLoop._serialize_config`

- **说明**: Serialize config to a JSON-compatible dict.
- **内部/外部调用**: `to_dict`, `asdict`

##### 方法: `RalphLoop._loop_type`

- **说明**: Label the loop for provenance and manifests.

##### 方法: `RalphLoop._phase2_features`

- **说明**: Phase 2 feature flags used by the current loop.

##### 方法: `RalphLoop._refresh_run_manifest`

- **说明**: Build and cache the current run manifest.
- **内部/外部调用**: `get_diagnostics`, `isoformat`, `_loop_type`, `_phase2_features`, `to_dict`, `keys`, `build_run_manifest`, `now`, `get`, `_serialize_config`

##### 方法: `RalphLoop._persist_run_manifest`

- **说明**: Write the current run manifest to disk and mirror it into the session.
- **内部/外部调用**: `dump`, `open`, `setdefault`, `mkdir`, `_refresh_run_manifest`

##### 方法: `RalphLoop._attach_factor_provenance`

- **说明**: Stamp provenance onto library factors that survived admission.
- **内部/外部调用**: `get`, `list_factors`, `_generator_family`, `reversed`, `to_dict`, `build_factor_provenance`, `_refresh_run_manifest`, `enumerate`

##### 方法: `RalphLoop._generator_family`

- **说明**: Return the active candidate generator label for provenance.

### 文件: `factorminer/core/session.py`

**模块说明**: Mining session management with persistence and resume support.

A ``MiningSession`` wraps the state that must survive across process
restarts: session metadata, per-iteration statistics, timing, and paths
to serialized artifacts (library, memory).

#### 类: `MiningSession`

**说明**: Manages a complete mining session with persistence.

##### 方法: `MiningSession.__post_init__`

- **内部/外部调用**: `now`, `isoformat`

##### 方法: `MiningSession.record_iteration`

- **说明**: Append iteration statistics to the session log.
- **内部/外部调用**: `isoformat`, `setdefault`, `append`, `now`

##### 方法: `MiningSession.total_iterations`

- **内部/外部调用**:

##### 方法: `MiningSession.last_library_size`

- **内部/外部调用**: `get`

##### 方法: `MiningSession.to_dict`

- **说明**: Serialize session state to a JSON-compatible dictionary.

##### 方法: `MiningSession.save`

- **说明**: Save session state to a JSON file.
- **内部/外部调用**: `dump`, `open`, `to_dict`, `mkdir`, `Path`

##### 方法: `MiningSession.load`

- **说明**: Load session from a JSON file.
- **内部/外部调用**: `cls`, `open`, `get`, `Path`, `load`

##### 方法: `MiningSession.get_summary`

- **说明**: Session summary statistics.
- **内部/外部调用**: `now`, `isoformat`, `fromisoformat`, `total_seconds`, `sum`, `get`

##### 方法: `MiningSession.finalize`

- **说明**: Mark the session as completed and record end time.
- **内部/外部调用**: `now`, `isoformat`

### 文件: `factorminer/core/types.py`

**模块说明**: Type system for the FactorMiner operator library.

Defines operator categories, signatures, specifications, and the canonical
set of raw market-data feature names used as leaf nodes in expression trees.

#### 类: `OperatorType`

**说明**: High-level category for every operator.

#### 类: `SignatureType`

**说明**: Describes how an operator maps inputs to outputs.

#### 类: `OperatorSpec`

**说明**: Immutable descriptor for a single operator in the library.

#### 函数: `_window_params`

- **说明**: Helper returning standard (window,) parameter triple.
- **内部/外部调用**:

#### 函数: `_build_operator_registry`

- **说明**: Construct the full operator registry.
- **内部/外部调用**: `_reg`, `wp`, `OperatorSpec`

#### 函数: `get_operator`

- **说明**: Look up an operator by name, raising ``KeyError`` if unknown.
- **内部/外部调用**: `sorted`, `keys`, `KeyError`

### 文件: `factorminer/data/__init__.py`

**模块说明**: FactorMiner data pipeline: loading, preprocessing, and tensor construction.

### 文件: `factorminer/data/loader.py`

**模块说明**: Market data loader supporting multiple formats and asset universes.

Loads OHLCV + amount data from CSV, Parquet, and HDF5 files. Supports
A-share universes (CSI500, CSI1000, HS300) and Binance crypto data.
Expected schema: datetime, asset_id, open, high, low, close, volume, amount.

The loader also accepts a small set of common aliases used by broker/data-vendor
exports, such as ``code``/``ticker`` for ``asset_id`` and ``amt`` for
``amount``.

#### 函数: `_infer_format`

- **内部/外部调用**: `lower`, `keys`, `get`

#### 函数: `_read_file`

- **说明**: Read a single data file into a DataFrame.
- **内部/外部调用**: `read_csv`, `read_parquet`, `read_hdf`

#### 函数: `_validate_columns`

- **说明**: Ensure required columns are present and normalise names.
- **内部/外部调用**: `lower`, `append`, `strip`, `get`, `rename`

#### 函数: `_coerce_types`

- **说明**: Ensure numeric types for OHLCV columns and datetime index.
- **内部/外部调用**: `astype`, `to_numeric`, `to_datetime`

#### 函数: `load_market_data`

- **说明**: Load market data from a single file.
- **内部/外部调用**: `_coerce_types`, `lower`, `reset_index`, `_read_file`, `exists`, `nunique`, `_validate_columns`, `info`, `sort_values`, `warning`, `get`, `Timestamp`, `_infer_format`, `FileNotFoundError`, `isin`, `Path`

#### 函数: `load_multiple`

- **说明**: Load and concatenate market data from multiple files.
- **内部/外部调用**: `reset_index`, `append`, `load_market_data`, `concat`, `sort_values`

#### 函数: `to_numpy`

- **说明**: Convert a DataFrame to a numpy array of the specified columns.
- **内部/外部调用**: `to_numpy`

### 文件: `factorminer/data/mock_data.py`

**模块说明**: Generate realistic synthetic market data for testing FactorMiner.

Produces multi-asset OHLCV data with:
- Volume clustering (GARCH-like)
- Volatility clustering
- Cross-sectional correlation via a common market factor
- Planted alpha signals for validating factor discovery
- OHLC consistency guarantees: low <= open,close <= high

#### 类: `MockConfig`

**说明**: Configuration for synthetic data generation.

#### 函数: `_bars_per_year`

- **说明**: Approximate number of bars in a trading year.

#### 函数: `_generate_timestamps`

- **说明**: Create a business-aware timestamp index.
- **内部/外部调用**: `replace`, `DatetimeIndex`, `extend`, `bdate_range`, `tolist`, `date_range`

#### 函数: `generate_mock_data`

- **说明**: Generate synthetic multi-asset OHLCV + amount data.
- **内部/外部调用**: `maximum`, `choice`, `reset_index`, `abs`, `sqrt`, `minimum`, `DataFrame`, `tolist`, `default_rng`, `append`, `log`, `_generate_timestamps`, `info`, `exp`, `cumsum`, `sort_values`, `max`, `_bars_per_year`, `astype`, `round`, `MockConfig`, `range`, `normal`, `empty`, `concat`

#### 函数: `generate_with_halts`

- **说明**: Generate mock data with simulated trading halts.
- **内部/外部调用**: `choice`, `generate_mock_data`, `default_rng`, `MockConfig`, `info`

### 文件: `factorminer/data/preprocessor.py`

**模块说明**: Data preprocessing pipeline for FactorMiner.

Handles derived feature computation, missing data imputation, trading halt
detection, cross-sectional standardisation, winsorisation, and quality checks.

#### 类: `PreprocessConfig`

**说明**: Configuration for the preprocessing pipeline.

#### 函数: `compute_vwap`

- **说明**: Add ``vwap`` column: amount / volume.  NaN when volume is zero.
- **内部/外部调用**: `where`, `copy`

#### 函数: `compute_returns`

- **说明**: Add ``returns`` column: close-to-close percentage change per asset.
- **内部/外部调用**: `copy`, `pct_change`, `groupby`, `sort_values`

#### 函数: `compute_derived_features`

- **说明**: Compute all derived features (vwap and returns).
- **内部/外部调用**: `compute_vwap`, `compute_returns`

#### 函数: `flag_halts`

- **说明**: Add boolean ``is_halt`` column.
- **内部/外部调用**: `sum`, `info`, `copy`

#### 函数: `mask_halts`

- **说明**: Set OHLCV and derived columns to NaN for halted bars.
- **内部/外部调用**: `copy`

#### 函数: `_extract_date`

- **说明**: Return the date component of a datetime series.

#### 函数: `fill_missing`

- **说明**: Fill missing values using a two-stage strategy.
- **内部/外部调用**: `copy`, `transform`, `fillna`, `drop`, `_extract_date`, `groupby`, `tolist`, `ffill`, `select_dtypes`

#### 函数: `winsorise`

- **说明**: Clip values in *columns* to the [lower, upper] percentile range
- **内部/外部调用**: `copy`, `transform`, `clip`, `any`, `notna`, `groupby`, `nanpercentile`

#### 函数: `cross_sectional_standardise`

- **说明**: Z-score standardise *columns* cross-sectionally at each time step.
- **内部/外部调用**: `replace`, `copy`, `transform`, `fillna`, `groupby`

#### 函数: `quality_check`

- **说明**: Drop time steps where the fraction of non-NaN values across assets
- **内部/外部调用**: `reset_index`, `apply`, `nunique`, `notna`, `groupby`, `sum`, `info`, `all`, `isin`

#### 函数: `preprocess`

- **说明**: Run the full preprocessing pipeline.
- **内部/外部调用**: `quality_check`, `mask_halts`, `compute_derived_features`, `cross_sectional_standardise`, `winsorise`, `flag_halts`, `info`, `PreprocessConfig`, `fill_missing`

### 文件: `factorminer/data/tensor_builder.py`

**模块说明**: Build the data tensor D in R^(M x T x F) for FactorMiner.

Converts preprocessed panel data into dense 3-D arrays indexed by
(assets, time_periods, features).  Supports numpy and optional torch backends.

#### 类: `TargetSpec`

**说明**: Definition of one aligned forward-return target.

##### 方法: `TargetSpec.column_name`


#### 类: `TensorConfig`

**说明**: Configuration for tensor construction.

#### 类: `TensorDataset`

**说明**: Container for the built tensor and associated metadata.

#### 函数: `compute_target`

- **说明**: Compute the target: next-bar open-to-close return.
- **内部/外部调用**: `compute_targets`, `TargetSpec`

#### 函数: `compute_targets`

- **说明**: Compute one or more named forward-return targets on the same panel.
- **内部/外部调用**: `_resolve_target_offsets`, `copy`, `log`, `shift`, `groupby`, `sort_values`

#### 函数: `_resolve_target_offsets`

- **说明**: Map a target spec to start/end price columns and offsets.
- **内部/外部调用**:

#### 函数: `_to_backend`

- **说明**: Convert a numpy array to the requested backend.
- **内部/外部调用**: `from_numpy`, `ImportError`, `astype`, `asarray`, `to`

#### 函数: `_build_3d`

- **说明**: Pivot panel data into a dense (M, T, F) numpy array.
- **内部/外部调用**: `map`, `copy`, `dropna`, `astype`, `full`, `to_numpy`, `enumerate`

#### 函数: `build_tensor`

- **说明**: Build a dense 3-D tensor from preprocessed panel data.
- **内部/外部调用**: `TensorConfig`, `next`, `_to_backend`, `iter`, `items`, `TensorDataset`, `unique`, `_build_3d`, `info`, `sort`, `get`

#### 函数: `temporal_split`

- **说明**: Split a :class:`TensorDataset` into train and test sets along time.
- **内部/外部调用**: `to_datetime`, `_slice`, `where`, `TensorDataset`, `items`, `Timestamp`, `arange`

#### 函数: `sample_assets`

- **说明**: Return a random subset of *m* assets from *ds*.
- **内部/外部调用**: `warning`, `choice`, `default_rng`, `TensorDataset`, `sort`, `items`

#### 函数: `build_pipeline`

- **说明**: End-to-end: compute target, build tensor, optionally split.
- **内部/外部调用**: `temporal_split`, `compute_target`, `TensorConfig`, `build_tensor`

### 文件: `factorminer/data/validation.py`

**模块说明**: File-level market data validation helpers.

#### 类: `ValidationIssue`

**说明**: One validation message emitted by the validator.

#### 类: `DataValidationReport`

**说明**: Structured validation result for a market-data file.

##### 方法: `DataValidationReport.valid_schema`


##### 方法: `DataValidationReport.has_warnings`

- **内部/外部调用**: `any`

##### 方法: `DataValidationReport.has_errors`

- **内部/外部调用**: `any`

##### 方法: `DataValidationReport.exit_code`


##### 方法: `DataValidationReport.to_dict`

- **内部/外部调用**: `max`, `asdict`, `_status_label`, `sum`, `items`

#### 函数: `validate_market_data`

- **说明**: Validate a market-data file against the canonical loader contract.
- **内部/外部调用**: `copy`, `isna`, `join`, `duplicated`, `drop_duplicates`, `nunique`, `to_numeric`, `_read_raw_frame`, `ValidationIssue`, `_count_non_monotonic_assets`, `append`, `rename`, `_find_leaky_columns`, `DataValidationReport`, `_count_ohlc_violations`, `dropna`, `sum`, `items`, `_build_canonical_mapping`, `_infer_format`, `mean`, `to_datetime`, `_count_negative_values`, `eq`, `strip`, `astype`, `Path`

#### 函数: `render_validation_report`

- **说明**: Render a human-readable validation summary.
- **内部/外部调用**: `join`, `append`, `items`, `_status_label`, `get`

#### 函数: `_infer_format`

- **内部/外部调用**: `lower`

#### 函数: `_read_raw_frame`

- **内部/外部调用**: `read_csv`, `read_parquet`, `read_hdf`

#### 函数: `_normalize`

- **内部/外部调用**: `lower`, `strip`

#### 函数: `_build_canonical_mapping`

- **内部/外部调用**: `add`, `_normalize`, `setdefault`, `append`, `get`

#### 函数: `_count_non_monotonic_assets`

- **内部/外部调用**: `dropna`, `to_datetime`, `groupby`

#### 函数: `_count_ohlc_violations`

- **内部/外部调用**: `sum`, `issubset`, `to_numeric`

#### 函数: `_count_negative_values`

- **内部/外部调用**: `sum`, `to_numeric`

#### 函数: `_find_leaky_columns`

- **内部/外部调用**: `_normalize`, `any`, `append`

#### 函数: `_status_label`


### 文件: `factorminer/evaluation/__init__.py`

**模块说明**: Multi-stage factor evaluation and validation pipeline.

### 文件: `factorminer/evaluation/admission.py`

**模块说明**: Admission rules for the factor library.

Implements the decision logic for whether a candidate factor should be
admitted to the library, replace an existing factor, or be rejected.

Admission Rule (Eq. 10):
    Admit alpha if |IC(alpha)| >= tau_IC  AND  max_{g in L} |rho(alpha, g)| < theta

Replacement Rule (Eq. 11):
    Replace g with alpha if:
        |IC(alpha)| >= 0.10  AND
        |IC(alpha)| >= 1.3 * |IC(g)|  AND
        |{g in L : |rho(alpha, g)| >= theta}| == 1

#### 类: `AdmissionDecision`

**说明**: Result of an admission check for a candidate factor.

#### 类: `StockThresholds`

**说明**: Default thresholds for A-share stock factor evaluation.

##### 方法: `StockThresholds.passes`

- **说明**: Check if a factor meets all stock-level thresholds.

#### 函数: `check_admission`

- **说明**: Standard admission check (Eq. 10).
- **内部/外部调用**: `AdmissionDecision`

#### 函数: `check_replacement`

- **说明**: Replacement admission check (Eq. 11).
- **内部/外部调用**: `AdmissionDecision`, `abs`, `items`, `get`

### 文件: `factorminer/evaluation/backtest.py`

**模块说明**: Full backtesting utilities for factor evaluation.

Provides time-series splitting, rolling and cumulative IC computation,
factor return attribution, and drawdown analysis.

#### 类: `SplitWindow`

**说明**: Indices for a single train/test split.

#### 类: `DrawdownResult`

**说明**: Results of drawdown analysis.

#### 函数: `train_test_split`

- **说明**: Simple contiguous train/test split.
- **内部/外部调用**: `SplitWindow`

#### 函数: `rolling_splits`

- **说明**: Generate rolling-window train/test splits.
- **内部/外部调用**: `SplitWindow`, `append`

#### 函数: `compute_ic_series`

- **说明**: Compute cross-sectional Spearman IC at each time step.
- **内部/外部调用**: `isfinite`, `spearmanr`, `full`, `range`, `sum`

#### 函数: `compute_rolling_ic`

- **说明**: Compute rolling-window average IC.
- **内部/外部调用**: `mean`, `isfinite`, `full`, `range`, `compute_ic_series`

#### 函数: `compute_cumulative_ic`

- **说明**: Compute cumulative (expanding-window) mean IC.
- **内部/外部调用**: `isfinite`, `full`, `range`, `compute_ic_series`

#### 函数: `compute_ic_stats`

- **说明**: Compute summary statistics for an IC series.
- **内部/外部调用**: `max`, `mean`, `isfinite`, `std`, `min`

#### 函数: `factor_return_attribution`

- **说明**: Attribute portfolio returns to individual factors.
- **内部/外部调用**: `max`, `mean`, `nanmean`, `isfinite`, `compute_ic_stats`, `full`, `range`, `compute_ic_series`, `sum`, `argsort`, `items`

#### 函数: `compute_drawdown`

- **说明**: Compute drawdown statistics from a cumulative return series.
- **内部/外部调用**: `DrawdownResult`, `append`, `accumulate`, `argmin`, `asarray`, `argmax`

#### 函数: `compute_sharpe_ratio`

- **说明**: Compute annualized Sharpe ratio.
- **内部/外部调用**: `mean`, `isfinite`, `sqrt`, `std`

#### 函数: `compute_calmar_ratio`

- **说明**: Compute Calmar ratio (annualized return / max drawdown).
- **内部/外部调用**: `mean`, `isfinite`, `abs`, `compute_drawdown`, `cumsum`

### 文件: `factorminer/evaluation/capacity.py`

**模块说明**: Capacity-aware backtesting for alpha factors.

Estimates market impact via a square-root model, evaluates net-of-cost
IC / ICIR, and determines the maximum capital that a factor can absorb
before its alpha degrades beyond acceptable limits.

#### 类: `CapacityConfig`

**说明**: Configuration for capacity-aware backtesting.

#### 类: `MarketImpactEstimate`

**说明**: Per-bar market impact estimate across the evaluation window.

#### 类: `CapacityEstimate`

**说明**: Result of a capacity sweep for a single factor.

#### 类: `NetCostResult`

**说明**: Net-of-cost evaluation at a specific capital level.

#### 类: `MarketImpactModel`

**说明**: Square-root market impact model.

##### 方法: `MarketImpactModel.__init__`

- **内部/外部调用**: `CapacityConfig`, `sqrt`

##### 方法: `MarketImpactModel.estimate_impact`

- **说明**: Estimate per-bar market impact for a given capital deployment.
- **内部/外部调用**: `sum`, `max`, `isnan`, `MarketImpactEstimate`, `mean`, `where`, `nanmean`, `sqrt`, `argpartition`, `min`, `full`, `range`, `empty`, `nanmax`, `enumerate`

#### 类: `CapacityEstimator`

**说明**: Evaluate factor capacity and net-of-cost performance.

##### 方法: `CapacityEstimator.__init__`

- **内部/外部调用**: `CapacityConfig`, `MarketImpactModel`

##### 方法: `CapacityEstimator._mean_ic`

- **说明**: Mean IC ignoring NaN.
- **内部/外部调用**: `isnan`, `mean`

##### 方法: `CapacityEstimator._net_returns`

- **说明**: Compute impact-adjusted returns.

##### 方法: `CapacityEstimator.estimate`

- **说明**: Run a capacity sweep across configured capital levels.
- **内部/外部调用**: `compute_ic`, `_mean_ic`, `abs`, `append`, `_net_returns`, `estimate_impact`, `CapacityEstimate`, `_interpolate_capacity`

##### 方法: `CapacityEstimator.net_cost_evaluation`

- **说明**: Evaluate a factor net of estimated market impact.
- **内部/外部调用**: `compute_ic`, `mean`, `nanmean`, `_net_returns`, `compute_icir`, `estimate_impact`, `NetCostResult`

##### 方法: `CapacityEstimator._interpolate_capacity`

- **说明**: Linearly interpolate the capital at which degradation hits *limit*.
- **内部/外部调用**: `abs`, `range`

### 文件: `factorminer/evaluation/causal.py`

**模块说明**: Causal validation layer for alpha factor candidates.

Provides Granger causality testing and intervention-based robustness
analysis to verify that discovered factors have genuine predictive
relationships with forward returns rather than spurious correlations.

Two complementary tests are combined into a single robustness score:

1. **Granger causality**: Does the factor signal Granger-cause returns
   after controlling for existing library factors?
2. **Intervention robustness**: Does factor IC remain stable under
   realistic data perturbations (volume shocks, volatility shocks,
   liquidity droughts)?

#### 类: `CausalConfig`

**说明**: Configuration for causal validation tests.

#### 类: `CausalTestResult`

**说明**: Result of causal validation for a single factor.

#### 类: `CausalValidator`

**说明**: Validates causal relationships between factor signals and returns.

##### 方法: `CausalValidator.__init__`

- **内部/外部调用**: `CausalConfig`, `RandomState`

##### 方法: `CausalValidator.validate`

- **说明**: Run causal validation on a single factor.
- **内部/外部调用**: `CausalTestResult`, `_compute_robustness_score`, `_granger_test`, `items`, `_intervention_test`

##### 方法: `CausalValidator.validate_batch`

- **说明**: Validate a batch of candidate factors.
- **内部/外部调用**: `validate`

##### 方法: `CausalValidator._granger_test`

- **说明**: Granger causality test for factor -> returns.
- **内部/外部调用**: `warning`, `max`, `_aggregate_top_assets`, `_run_granger_multivariate`, `_run_granger_bivariate`, `min`, `_is_degenerate`

##### 方法: `CausalValidator._run_granger_bivariate`

- **说明**: Bivariate Granger test using statsmodels.
- **内部/外部调用**: `simplefilter`, `grangercausalitytests`, `min`, `range`, `catch_warnings`, `column_stack`, `get`

##### 方法: `CausalValidator._run_granger_multivariate`

- **说明**: Multivariate Granger via VAR, controlling for library factors.
- **内部/外部调用**: `simplefilter`, `_aggregate_top_assets`, `warning`, `test_causality`, `isnan`, `fit`, `append`, `min`, `_pca_reduce`, `any`, `VAR`, `catch_warnings`, `column_stack`, `items`, `_is_degenerate`

##### 方法: `CausalValidator._intervention_test`

- **说明**: Intervention-based robustness test.
- **内部/外部调用**: `warning`, `compute_ic`, `mean`, `abs`, `append`, `isnan`, `_build_intervention_scenarios`

##### 方法: `CausalValidator._build_intervention_scenarios`

- **说明**: Construct the three intervention scenarios.
- **内部/外部调用**: `choice`, `max`, `copy`, `append`, `nanstd`, `randn`, `randint`, `enumerate`

##### 方法: `CausalValidator._compute_robustness_score`

- **说明**: Combine Granger and intervention results into a 0-1 score.
- **内部/外部调用**: `clip`, `min`

##### 方法: `CausalValidator._aggregate_top_assets`

- **说明**: Average across the top-k assets (by mean absolute value) to
- **内部/外部调用**: `simplefilter`, `where`, `nanmean`, `abs`, `argpartition`, `min`, `catch_warnings`, `isnan`

##### 方法: `CausalValidator._is_degenerate`

- **说明**: Check if a series is constant or all-NaN.
- **内部/外部调用**: `isnan`, `std`

##### 方法: `CausalValidator._pca_reduce`

- **说明**: Reduce columns of X via truncated SVD (no sklearn dependency).
- **内部/外部调用**: `warning`, `where`, `nanmean`, `svd`, `min`, `isnan`

### 文件: `factorminer/evaluation/combination.py`

**模块说明**: Factor combination strategies for building composite signals.

Implements Equal-Weight, IC-Weighted, and Orthogonal combination methods
for merging multiple alpha factors into a single composite signal, following
the methodology described in the FactorMiner paper.

#### 类: `FactorCombiner`

**说明**: Combine multiple factor signals into a single composite signal.

##### 方法: `FactorCombiner.equal_weight`

- **说明**: Equal-Weight (EW): simple average of cross-sectionally standardized factors.
- **内部/外部调用**: `values`, `nanmean`, `stack`, `_cross_sectional_standardize`

##### 方法: `FactorCombiner.ic_weighted`

- **说明**: IC-Weighted (ICW): weight factors proportionally by their historical IC.
- **内部/外部调用**: `get`, `zeros`, `values`, `isnan`, `next`, `where`, `isfinite`, `iter`, `keys`, `sum`, `_cross_sectional_standardize`, `items`, `equal_weight`

##### 方法: `FactorCombiner.orthogonal`

- **说明**: Orthogonal: Gram-Schmidt orthogonalization before averaging.
- **内部/外部调用**: `values`, `nanmean`, `stack`, `_cross_sectional_standardize`, `_gram_schmidt`

##### 方法: `FactorCombiner._cross_sectional_standardize`

- **说明**: Standardize signals cross-sectionally (across assets) at each time step.
- **内部/外部调用**: `nanstd`, `where`, `nanmean`, `asarray`

##### 方法: `FactorCombiner._gram_schmidt`

- **说明**: Modified Gram-Schmidt orthogonalization on flattened factor vectors.
- **内部/外部调用**: `copy`, `reshape`, `where`, `zip`, `append`, `dot`, `ravel`, `isnan`, `enumerate`

### 文件: `factorminer/evaluation/correlation.py`

**模块说明**: Efficient correlation computation for factor evaluation.

Provides batch Spearman rank correlation, vectorized cross-sectional
correlation, and incremental correlation matrix updates for the
factor library.  Supports both numpy and optional torch backends.

#### 类: `IncrementalCorrelationMatrix`

**说明**: Maintains a correlation matrix that can be incrementally updated.

##### 方法: `IncrementalCorrelationMatrix.__init__`


##### 方法: `IncrementalCorrelationMatrix.size`

- **内部/外部调用**:

##### 方法: `IncrementalCorrelationMatrix.factor_ids`

- **内部/外部调用**:

##### 方法: `IncrementalCorrelationMatrix._compute_pair_corr`

- **说明**: Compute average cross-sectional Spearman between two factors.
- **内部/外部调用**: `mean`, `sqrt`, `range`, `sum`, `isnan`

##### 方法: `IncrementalCorrelationMatrix.add_factor`

- **说明**: Add a factor and compute its correlation with all existing factors.
- **内部/外部调用**: `max`, `append`, `_compute_pair_corr`, `_rank_columns`, `min`

##### 方法: `IncrementalCorrelationMatrix.remove_factor`

- **说明**: Remove a factor from the matrix.
- **内部/外部调用**: `pop`

##### 方法: `IncrementalCorrelationMatrix.get_correlation`

- **说明**: Get cached correlation between two factors.
- **内部/外部调用**: `min`, `max`

##### 方法: `IncrementalCorrelationMatrix.get_max_correlation`

- **说明**: Get the maximum absolute correlation of a factor with all others.
- **内部/外部调用**: `get_correlation`, `abs`

##### 方法: `IncrementalCorrelationMatrix.to_matrix`

- **说明**: Return the full correlation matrix as a numpy array.
- **内部/外部调用**: `eye`, `range`, `get_correlation`

#### 函数: `_rank_columns`

- **说明**: Rank each column of x independently, leaving NaN as NaN.
- **内部/外部调用**: `rankdata`, `full_like`, `range`, `sum`, `isnan`

#### 函数: `batch_spearman_correlation`

- **说明**: Compute Spearman correlation between one candidate and multiple library factors.
- **内部/外部调用**: `zeros`, `mean`, `sqrt`, `_rank_columns`, `array`, `range`, `sum`, `isnan`

#### 函数: `batch_spearman_pairwise`

- **说明**: Compute pairwise Spearman correlation matrix for a list of signal arrays.
- **内部/外部调用**: `sum`, `reshape`, `mean`, `sqrt`, `eye`, `_rank_columns`, `range`, `array`, `isnan`

#### 函数: `_try_torch_rank_correlation`

- **说明**: Attempt to compute rank correlations using PyTorch for GPU acceleration.
- **内部/外部调用**: `numpy`, `zeros`, `from_numpy`, `max`, `mean`, `cpu`, `sqrt`, `to`, `range`, `sum`, `is_available`, `device`, `isnan`, `argsort`

#### 函数: `compute_correlation_batch`

- **说明**: Compute correlations between candidate and library, with backend selection.
- **内部/外部调用**: `_try_torch_rank_correlation`, `batch_spearman_correlation`

### 文件: `factorminer/evaluation/metrics.py`

**模块说明**: Core evaluation metrics for alpha factors.

Provides vectorized, production-quality implementations of Information
Coefficient (IC), ICIR, quintile analysis, turnover, and comprehensive
factor statistics used by the validation pipeline.

#### 函数: `compute_ic`

- **说明**: Compute IC_t = Corr_rank(s_t, r_{t+1}) for each time period.
- **内部/外部调用**: `rankdata`, `mean`, `sqrt`, `full`, `range`, `sum`, `isnan`

#### 函数: `compute_ic_vectorized`

- **说明**: Fully vectorized IC computation (faster for large M, T).
- **内部/外部调用**: `rankdata`, `mean`, `where`, `sqrt`, `full`, `range`, `sum`, `isnan`

#### 函数: `compute_icir`

- **说明**: Compute ICIR = mean(IC) / std(IC).
- **内部/外部调用**: `mean`, `std`, `isnan`

#### 函数: `compute_ic_mean`

- **说明**: Compute mean absolute IC.
- **内部/外部调用**: `mean`, `abs`, `isnan`

#### 函数: `compute_ic_win_rate`

- **说明**: Fraction of periods with positive IC.
- **内部/外部调用**: `isnan`, `mean`

#### 函数: `compute_pairwise_correlation`

- **说明**: Time-averaged cross-sectional Spearman correlation between two factors.
- **内部/外部调用**: `rankdata`, `mean`, `sqrt`, `append`, `range`, `sum`, `isnan`

#### 函数: `compute_quintile_returns`

- **说明**: Sort assets into quintiles by factor signal, compute average returns.
- **内部/外部调用**: `sum`, `rankdata`, `ceil`, `mean`, `sqrt`, `std`, `append`, `clip`, `astype`, `any`, `range`, `array`, `isnan`, `arange`

#### 函数: `compute_turnover`

- **说明**: Compute average portfolio turnover rate.
- **内部/外部调用**: `max`, `mean`, `where`, `append`, `argpartition`, `range`, `sum`, `isnan`

#### 函数: `compute_factor_stats`

- **说明**: Compute comprehensive factor statistics.
- **内部/外部调用**: `compute_ic`, `mean`, `compute_turnover`, `compute_ic_mean`, `compute_ic_win_rate`, `update`, `std`, `compute_icir`, `compute_quintile_returns`, `sum`, `isnan`

### 文件: `factorminer/evaluation/pipeline.py`

**模块说明**: Multi-stage factor evaluation and validation pipeline.

Implements Algorithm 1 Step 3: the four-stage evaluation cascade that
screens, deduplicates, and validates candidate alpha factors before
admitting them to the factor library.

Stages:
    1. Fast IC screening on a subset of assets
    2. Correlation check against the existing library
    2.5. Replacement check for rejected-but-strong candidates
    3. Intra-batch deduplication
    4. Full validation on the complete asset universe

Supports parallel evaluation via a configurable multiprocessing worker pool.

#### 类: `CandidateFactor`

**说明**: A candidate factor to be evaluated.

#### 类: `EvaluationResult`

**说明**: Result of evaluating a single candidate through the pipeline.

##### 方法: `EvaluationResult.to_trajectory_dict`

- **说明**: Convert to a dict compatible with the memory formation trajectory format.

#### 类: `FactorLibraryView`

**说明**: Read-only view of the factor library for the pipeline.

##### 方法: `FactorLibraryView.size`

- **内部/外部调用**:

##### 方法: `FactorLibraryView.get_signals_tensor`

- **说明**: Return library signals as a (N, M, T) tensor.
- **内部/外部调用**: `array`, `stack`, `reshape`

#### 类: `PipelineConfig`

**说明**: Configuration for the validation pipeline.

##### 方法: `PipelineConfig.from_config`

- **说明**: Build from MiningConfig and EvaluationConfig objects.
- **内部/外部调用**: `cls`

#### 类: `ValidationPipeline`

**说明**: Multi-stage factor evaluation pipeline.

##### 方法: `ValidationPipeline.__init__`

- **内部/外部调用**: `arange`, `choice`, `default_rng`

##### 方法: `ValidationPipeline.evaluate_batch`

- **说明**: Run the full multi-stage evaluation on a batch of candidates.
- **内部/外部调用**: `values`, `_ensure_signals`, `_stage1_ic_screen`, `_stage2_correlation_check`, `append`, `_stage4_full_validation`, `_stage25_replacement_check`, `_stage3_batch_dedup`, `sum`, `info`

##### 方法: `ValidationPipeline._ensure_signals`

- **说明**: Compute signals for candidates that don't have them yet.
- **内部/外部调用**: `compute_signals_fn`

##### 方法: `ValidationPipeline._stage1_ic_screen`

- **说明**: Stage 1: Fast IC screening on asset subset.
- **内部/外部调用**: `compute_ic`, `EvaluationResult`, `mean`, `abs`, `append`, `isnan`

##### 方法: `ValidationPipeline._stage2_correlation_check`

- **说明**: Stage 2: Correlation check against the library.
- **内部/外部调用**: `EvaluationResult`, `get_signals_tensor`, `abs`, `append`, `compute_correlation_batch`, `argmax`, `get`, `enumerate`

##### 方法: `ValidationPipeline._stage25_replacement_check`

- **说明**: Stage 2.5: Check if rejected candidates can replace library members.
- **内部/外部调用**: `EvaluationResult`, `check_replacement`, `get`, `append`

##### 方法: `ValidationPipeline._stage3_batch_dedup`

- **说明**: Stage 3: Intra-batch deduplication.
- **内部/外部调用**: `add`, `batch_spearman_pairwise`, `EvaluationResult`, `sorted`, `abs`, `append`, `range`, `get`

##### 方法: `ValidationPipeline._stage4_full_validation`

- **说明**: Stage 4: Full validation on complete asset universe.
- **内部/外部调用**: `_stage4_parallel`, `append`, `_validate_single`

##### 方法: `ValidationPipeline._validate_single`

- **说明**: Run full validation for a single candidate.
- **内部/外部调用**: `EvaluationResult`, `compute_factor_stats`, `get`

##### 方法: `ValidationPipeline._stage4_parallel`

- **说明**: Run stage 4 in parallel using ProcessPoolExecutor.
- **内部/外部调用**: `EvaluationResult`, `ProcessPoolExecutor`, `submit`, `append`, `as_completed`, `error`, `get`, `result`

#### 函数: `_evaluate_single_candidate_ic`

- **说明**: Compute IC series, IC mean, and ICIR for a single candidate.
- **内部/外部调用**: `compute_ic`, `mean`, `abs`, `compute_icir`, `isnan`

#### 函数: `run_evaluation_pipeline`

- **说明**: One-shot convenience function to run the full evaluation pipeline.
- **内部/外部调用**: `evaluate_batch`, `ValidationPipeline`

### 文件: `factorminer/evaluation/portfolio.py`

**模块说明**: Portfolio construction and quintile backtesting.

Implements quintile-sorted long-short portfolio backtesting with
transaction cost pressure testing, following the FactorMiner paper methodology.

#### 类: `PortfolioBacktester`

**说明**: Backtest factor signals using quintile portfolios.

##### 方法: `PortfolioBacktester.quintile_backtest`

- **说明**: Run quintile portfolio backtest.
- **内部/外部调用**: `mean`, `where`, `nanmean`, `_rank_array`, `isfinite`, `spearmanr`, `compute_turnover`, `nancumsum`, `linspace`, `std`, `full`, `range`, `sum`, `asarray`

##### 方法: `PortfolioBacktester.cost_pressure_test`

- **说明**: Run backtest under multiple transaction cost settings (in bps).
- **内部/外部调用**: `quintile_backtest`

##### 方法: `PortfolioBacktester.compute_turnover`

- **说明**: Compute daily turnover of the top/bottom quintile portfolios.
- **内部/外部调用**: `zeros`, `max`, `where`, `isfinite`, `argpartition`, `range`, `sum`, `asarray`

#### 函数: `_rank_array`

- **说明**: Compute percentile ranks in [0, 1] for a 1-D array.
- **内部/外部调用**: `max`, `copy`, `range`, `empty`, `argsort`, `arange`

### 文件: `factorminer/evaluation/regime.py`

**模块说明**: Regime-aware factor validation.

Classifies market periods into BULL / BEAR / SIDEWAYS regimes using
rolling return and volatility statistics, then evaluates factor IC
within each regime to ensure robustness across market conditions.

#### 类: `MarketRegime`

**说明**: Market regime labels.

#### 类: `RegimeConfig`

**说明**: Parameters controlling regime detection and per-regime IC validation.

#### 类: `RegimeClassification`

**说明**: Output of :class:`RegimeDetector.classify`.

#### 类: `RegimeDetector`

**说明**: Classify time periods into market regimes.

##### 方法: `RegimeDetector.__init__`

- **内部/外部调用**: `RegimeConfig`

##### 方法: `RegimeDetector.classify`

- **说明**: Classify each period into a market regime.
- **内部/外部调用**: `sum`, `RegimeClassification`, `mean`, `nanmean`, `_rolling_nanstd`, `std`, `full`, `percentile`, `_rolling_nanmean`, `isnan`

##### 方法: `RegimeDetector._rolling_nanmean`

- **说明**: Rolling mean that ignores NaN, returning NaN for the first *window-1* entries.
- **内部/外部调用**: `mean`, `full`, `range`, `isnan`

##### 方法: `RegimeDetector._rolling_nanstd`

- **说明**: Rolling std (ddof=1) that ignores NaN.
- **内部/外部调用**: `std`, `full`, `range`, `isnan`

#### 类: `RegimeICResult`

**说明**: Evaluation result for a single factor across market regimes.

#### 类: `RegimeAwareEvaluator`

**说明**: Evaluate factor IC within each market regime.

##### 方法: `RegimeAwareEvaluator.__init__`

- **内部/外部调用**: `RegimeConfig`

##### 方法: `RegimeAwareEvaluator.evaluate`

- **说明**: Evaluate a single factor across regimes.
- **内部/外部调用**: `mean`, `where`, `abs`, `RegimeICResult`, `_compute_icir`, `_compute_ic`, `sum`, `isnan`

##### 方法: `RegimeAwareEvaluator.evaluate_batch`

- **说明**: Evaluate multiple factors.
- **内部/外部调用**: `items`, `evaluate`

##### 方法: `RegimeAwareEvaluator._compute_ic`

- **说明**: Cross-sectional Spearman IC per period.
- **内部/外部调用**: `rankdata`, `mean`, `sqrt`, `full`, `range`, `sum`, `isnan`

##### 方法: `RegimeAwareEvaluator._compute_icir`

- **说明**: ICIR = mean(IC) / std(IC).
- **内部/外部调用**: `mean`, `std`

#### 类: `TrendRegime`

#### 类: `VolRegime`

#### 类: `MeanRevRegime`

#### 类: `RegimeState`

**说明**: Composite regime state: trend + vol + mean-reversion classification.

##### 方法: `RegimeState.to_dict`


##### 方法: `RegimeState.from_dict`

- **内部/外部调用**: `cls`, `MeanRevRegime`, `VolRegime`, `TrendRegime`, `get`

##### 方法: `RegimeState.__str__`


##### 方法: `RegimeState.label`

- **内部/外部调用**:

#### 类: `StreamingRegimeConfig`

**说明**: Configuration for StreamingRegimeDetector.

#### 类: `StreamingRegimeDetector`

**说明**: Bar-by-bar O(1) regime classifier using exponentially-weighted stats.

##### 方法: `StreamingRegimeDetector.__init__`

- **内部/外部调用**: `RLock`, `deque`, `RegimeState`, `StreamingRegimeConfig`

##### 方法: `StreamingRegimeDetector.update`

- **说明**: Process one bar and return updated RegimeState.
- **内部/外部调用**: `_update_moments`, `nanmean`, `_apply_smoothing`, `append`, `nanstd`, `_record_transition`, `_classify`

##### 方法: `StreamingRegimeDetector.get_current_regime`


##### 方法: `StreamingRegimeDetector.get_regime_history`

- **内部/外部调用**:

##### 方法: `StreamingRegimeDetector.regime_transition_probability`

- **说明**: Return dict of 'from/to' → empirical probability.
- **内部/外部调用**: `values`, `sum`, `items`

##### 方法: `StreamingRegimeDetector.reset`

- **内部/外部调用**: `__init__`

##### 方法: `StreamingRegimeDetector._update_moments`

- **内部/外部调用**: `pop`, `append`

##### 方法: `StreamingRegimeDetector._classify`

- **内部/外部调用**: `_classify_vol`, `_classify_mean_rev`, `_classify_trend`, `RegimeState`

##### 方法: `StreamingRegimeDetector._classify_trend`

- **内部/外部调用**: `max`, `sqrt`

##### 方法: `StreamingRegimeDetector._classify_vol`

- **内部/外部调用**: `max`, `quantile`, `sqrt`, `array`

##### 方法: `StreamingRegimeDetector._classify_mean_rev`

- **内部/外部调用**: `mean`, `var`, `append`, `array`, `diff`

##### 方法: `StreamingRegimeDetector._apply_smoothing`

- **说明**: HMM-inspired: resist single-bar flips via smoothing weight.
- **内部/外部调用**: `random`

##### 方法: `StreamingRegimeDetector._record_transition`

- **内部/外部调用**: `get`, `label`

### 文件: `factorminer/evaluation/report_viewer.py`

**模块说明**: Static report generation for FactorMiner artifacts.

This module turns persisted mining artifacts into a local markdown or HTML
report without introducing a web dashboard. It understands the canonical
factor library JSON, optional session logs, and optional benchmark payloads.

#### 类: `ReportSection`

**说明**: One benchmark payload section in the final report.

#### 函数: `_resolve_json_path`

- **内部/外部调用**: `with_suffix`, `FileNotFoundError`, `exists`, `is_dir`, `Path`

#### 函数: `_load_json_source`

- **内部/外部调用**: `_resolve_json_path`, `read_text`, `loads`

#### 函数: `_json_label`

- **内部/外部调用**: `is_dir`, `Path`

#### 函数: `_first_non_empty`

- **内部/外部调用**: `strip`

#### 函数: `_fmt_num`

- **内部/外部调用**:

#### 函数: `_table_cell`

- **内部/外部调用**:

#### 函数: `_factor_rows`

- **内部/外部调用**: `append`, `_first_non_empty`, `isdigit`, `sort`, `get`

#### 函数: `_session_counts`

- **内部/外部调用**: `strip`, `most_common`, `Counter`, `sum`, `get`

#### 函数: `_benchmark_sections`

- **内部/外部调用**: `get`, `_load_json_source`, `ReportSection`, `append`, `_json_label`, `_looks_like_suite_payload`, `items`

#### 函数: `_looks_like_suite_payload`

- **内部/外部调用**: `values`, `any`, `all`

#### 函数: `_benchmark_universe_rows`

- **内部/外部调用**: `get`, `append`, `_first_numeric`, `items`

#### 函数: `_first_numeric`

- **内部/外部调用**: `get`

#### 函数: `build_report_payload`

- **说明**: Load and normalize report inputs into a structured payload.
- **内部/外部调用**: `_session_counts`, `_benchmark_universe_rows`, `isoformat`, `now`, `_load_json_source`, `_factor_rows`, `extend`, `_json_label`, `most_common`, `_benchmark_sections`, `Counter`, `get`

#### 函数: `_markdown_table`

- **内部/外部调用**: `join`, `_table_cell`

#### 函数: `_html_table`

- **内部/外部调用**: `join`, `escape`, `append`, `_html_cell_value`

#### 函数: `_html_cell_value`

- **内部/外部调用**:

#### 函数: `render_markdown_report`

- **内部/外部调用**: `_markdown_table`, `rstrip`, `join`, `append`, `_fmt_num`, `get`

#### 函数: `render_html_report`

- **内部/外部调用**: `join`, `_html_table`, `_card`, `extend`, `escape`, `append`, `_fmt_num`, `get`

#### 函数: `_card`

- **内部/外部调用**: `_html_cell_value`, `escape`

#### 函数: `generate_report`

- **说明**: Generate a static report and optionally persist it to disk.
- **内部/外部调用**: `render_markdown_report`, `build_report_payload`, `render_html_report`, `mkdir`, `write_text`, `Path`

#### 函数: `write_report`

- **说明**: Generate a report and save it to ``output_path``.
- **内部/外部调用**: `generate_report`, `Path`

#### 函数: `main`

- **说明**: CLI entry point for local report generation.
- **内部/外部调用**: `add_argument`, `parse_args`, `generate_report`, `ArgumentParser`

### 文件: `factorminer/evaluation/research.py`

**模块说明**: Research-first multi-horizon scoring and model evaluation.

#### 类: `FactorGeometryDiagnostics`

**说明**: How much new information a factor adds beyond the current library.

#### 类: `FactorScoreVector`

**说明**: Multi-horizon quality summary used in research mode.

##### 方法: `FactorScoreVector.to_dict`

- **内部/外部调用**: `asdict`

#### 函数: `compute_factor_geometry`

- **说明**: Compute soft library geometry metrics for a candidate.
- **内部/外部调用**: `max`, `compute_factor_stats`, `column_stack`, `mean`, `abs`, `compute_pairwise_correlation`, `append`, `FactorGeometryDiagnostics`, `_effective_rank`, `nanstd`, `var`, `lstsq`, `_unflatten_panel`, `_flatten_panel`

#### 函数: `build_score_vector`

- **说明**: Aggregate per-target metrics into one research-mode score vector.
- **内部/外部调用**: `get`, `values`, `max`, `_decay_slope`, `mean`, `_normalized_weights`, `keys`, `_cross_horizon_consistency`, `FactorScoreVector`, `min`, `array`, `sum`, `asarray`, `items`, `_bootstrap_standard_error`

#### 函数: `passes_research_admission`

- **说明**: Apply research-mode admission rules on top of paper-style correlation.

#### 函数: `run_research_model_suite`

- **说明**: Fit research-mode models on rolling windows and report net IR/stability.
- **内部/外部调用**: `PortfolioBacktester`, `mean`, `rolling_splits`, `setdefault`, `_selection_stability`, `append`, `_composite_regime_report`, `FactorSelector`, `_series_ir`, `quintile_backtest`, `_fit_research_model`, `_weighted_composite`, `items`

#### 函数: `_fit_research_model`

- **内部/外部调用**: `forward_stepwise`, `max`, `xgboost_selection`, `fit`, `_prepare_panel`, `abs`, `ElasticNetCV`, `logspace`, `min`, `RidgeCV`, `lasso_selection`, `items`, `enumerate`

#### 函数: `_weighted_composite`

- **内部/外部调用**: `sum`, `values`, `next`, `nanmean`, `zeros_like`, `iter`, `abs`, `ones_like`, `where`, `nanstd`, `astype`, `array`, `isnan`, `enumerate`

#### 函数: `_bootstrap_standard_error`

- **内部/外部调用**: `BootstrapICTester`, `isfinite`, `SignificanceConfig`, `compute_ci`

#### 函数: `_normalized_weights`

- **内部/外部调用**: `sum`, `max`, `array`, `get`, `enumerate`

#### 函数: `_decay_slope`

- **内部/外部调用**: `polyfit`, `std`, `array`, `items`

#### 函数: `_cross_horizon_consistency`

- **内部/外部调用**: `values`, `mean`, `abs`, `sign`, `sum`

#### 函数: `_flatten_panel`

- **内部/外部调用**: `reshape`, `where`, `nanmean`, `isfinite`, `nanstd`, `asarray`

#### 函数: `_unflatten_panel`

- **内部/外部调用**: `full`, `reshape`

#### 函数: `_effective_rank`

- **内部/外部调用**: `svd`, `log`, `min`, `sum`, `exp`

#### 函数: `_selection_stability`

- **内部/外部调用**: `mean`, `append`, `range`

#### 函数: `_series_ir`

- **内部/外部调用**: `mean`, `isfinite`, `std`, `asarray`

#### 函数: `_composite_regime_report`

- **内部/外部调用**: `RegimeConfig`, `PortfolioBacktester`, `evaluate`, `RegimeAwareEvaluator`, `_series_ir`, `RegimeDetector`, `classify`, `quintile_backtest`, `items`

### 文件: `factorminer/evaluation/runtime.py`

**模块说明**: Shared runtime evaluation helpers for strict factor recomputation.

#### 类: `SignalComputationError`

**说明**: Raised when a factor cannot be recomputed under strict policies.

#### 类: `DatasetSplit`

**说明**: One temporal view into the evaluation dataset.

##### 方法: `DatasetSplit.size`

- **内部/外部调用**:

##### 方法: `DatasetSplit.get_target`


#### 类: `EvaluationDataset`

**说明**: Canonical dataset used for analysis commands.

##### 方法: `EvaluationDataset.get_split`

- **内部/外部调用**: `KeyError`

##### 方法: `EvaluationDataset.get_target`


#### 类: `FactorEvaluationArtifact`

**说明**: Recomputed signals and metrics for one factor.

##### 方法: `FactorEvaluationArtifact.succeeded`


#### 函数: `load_runtime_dataset`

- **说明**: Load raw market data into a canonical evaluation dataset.
- **内部/外部调用**: `copy`, `reset_index`, `_resolve_target_specs`, `merge`, `preprocess`, `build_tensor`, `DatasetSplit`, `_resolve_feature_columns`, `sort_values`, `_build_named_split`, `enumerate`, `arange`, `EvaluationDataset`, `TensorConfig`, `_target_column_for_name`, `to_datetime`, `compute_targets`, `to_numpy`, `asarray`

#### 函数: `evaluate_factors`

- **说明**: Recompute factor signals and metrics across all dataset splits.
- **内部/外部调用**: `FactorEvaluationArtifact`, `compute_factor_stats`, `isnan`, `all`, `append`, `try_parse`, `items`, `asarray`, `compute_tree_signals`, `get_target`

#### 函数: `compute_tree_signals`

- **说明**: Evaluate an expression tree under an explicit failure policy.
- **内部/外部调用**: `_handle_signal_failure`, `evaluate`, `to_string`, `all`, `asarray`, `isnan`, `SignalComputationError`

#### 函数: `compute_correlation_matrix`

- **说明**: Compute a true pairwise factor correlation matrix on one split.
- **内部/外部调用**: `zeros`, `range`, `compute_pairwise_correlation`

#### 函数: `select_top_k`

- **说明**: Sort succeeded artifacts by split abs-IC and return the top-k subset.
- **内部/外部调用**: `sort`, `abs`, `get`

#### 函数: `summarize_failures`

- **说明**: Return human-readable failure summaries.

#### 函数: `resolve_split_for_fit_eval`

- **说明**: Map fit/eval CLI period values to runtime split names.

#### 函数: `analysis_split_names`

- **说明**: Map analysis CLI period values to one or two runtime split names.

#### 函数: `_resolve_feature_columns`

- **内部/外部调用**: `keys`, `lstrip`, `append`

#### 函数: `_build_named_split`

- **内部/外部调用**: `to_datetime`, `where`, `DatasetSplit`, `items`, `Timestamp`

#### 函数: `_resolve_target_specs`

- **内部/外部调用**: `TargetSpec`, `get`

#### 函数: `_target_column_for_name`


#### 函数: `_handle_signal_failure`

- **内部/外部调用**: `warning`, `generate_synthetic_signals`, `SignalComputationError`

#### 函数: `generate_synthetic_signals`

- **说明**: Deterministic pseudo-signals for demo/mock workflows.
- **内部/外部调用**: `random`, `RandomState`, `astype`, `randn`, `hash`

### 文件: `factorminer/evaluation/selection.py`

**模块说明**: Factor selection methods for identifying sparse, high-value subsets.

Implements Lasso (L1), Forward Stepwise, and XGBoost-based selection
strategies for choosing an optimal subset of factors from the mined library.

#### 类: `FactorSelector`

**说明**: Select optimal factor subsets from the factor library.

##### 方法: `FactorSelector.lasso_selection`

- **说明**: Lasso: L1-regularized linear regression for factor selection.
- **内部/外部调用**: `Lasso`, `fit`, `LassoCV`, `_prepare_panel`, `abs`, `append`, `info`, `sort`, `enumerate`

##### 方法: `FactorSelector.forward_stepwise`

- **说明**: Forward Stepwise: greedy selection maximizing combined ICIR.
- **内部/外部调用**: `_composite_icir`, `keys`, `append`, `min`, `range`, `info`, `discard`

##### 方法: `FactorSelector.xgboost_selection`

- **说明**: XGBoost: gradient boosting for nonlinear factor interactions.
- **内部/外部调用**: `fit`, `_prepare_panel`, `XGBRegressor`, `range`, `sort`

##### 方法: `FactorSelector._prepare_panel`

- **说明**: Flatten panel data to (samples, features) for sklearn-style models.
- **内部/外部调用**: `values`, `next`, `sorted`, `iter`, `isfinite`, `keys`, `empty`, `ravel`, `all`, `column_stack`

##### 方法: `FactorSelector._composite_icir`

- **说明**: Compute ICIR of the equal-weight composite of selected factors.
- **内部/外部调用**: `mean`, `nanmean`, `where`, `isfinite`, `stack`, `spearmanr`, `std`, `nanstd`, `astype`, `append`, `full`, `range`, `sum`

### 文件: `factorminer/evaluation/significance.py`

**模块说明**: Statistical significance testing for alpha factors.

Provides block bootstrap confidence intervals, Benjamini-Hochberg FDR
control, and Deflated Sharpe Ratio (Bailey & López de Prado, 2014) to
guard against data-snooping and multiple-testing bias in factor research.

#### 类: `SignificanceConfig`

**说明**: Configuration for all significance tests.

#### 类: `BootstrapCIResult`

**说明**: Result of a block bootstrap confidence interval for mean |IC|.

#### 类: `BootstrapICTester`

**说明**: Block bootstrap tester for IC series significance.

##### 方法: `BootstrapICTester.__init__`

- **内部/外部调用**: `RandomState`

##### 方法: `BootstrapICTester.compute_ci`

- **说明**: Compute block-bootstrap CI for mean |IC|.
- **内部/外部调用**: `BootstrapCIResult`, `mean`, `abs`, `std`, `_block_bootstrap_means`, `percentile`, `isnan`

##### 方法: `BootstrapICTester.compute_p_value`

- **说明**: Estimate a two-sided p-value for non-zero mean IC.
- **内部/外部调用**: `sum`, `choice`, `mean`, `abs`, `range`, `empty`, `isnan`

##### 方法: `BootstrapICTester._effective_block_size`

- **说明**: Adaptive block size: min(configured, T // 10), at least 1.
- **内部/外部调用**: `max`, `min`

##### 方法: `BootstrapICTester._block_bootstrap_means`

- **说明**: Generate bootstrap distribution of the sample mean.
- **内部/外部调用**: `_effective_block_size`, `ceil`, `mean`, `concatenate`, `range`, `randint`, `empty`, `arange`

#### 类: `FDRResult`

**说明**: Result of Benjamini-Hochberg FDR correction.

#### 类: `FDRController`

**说明**: Benjamini-Hochberg FDR correction for multiple factor testing.

##### 方法: `FDRController.__init__`


##### 方法: `FDRController.apply_fdr`

- **说明**: Apply Benjamini-Hochberg procedure.
- **内部/外部调用**: `values`, `keys`, `empty`, `min`, `array`, `range`, `sum`, `argsort`, `FDRResult`, `enumerate`, `arange`

##### 方法: `FDRController.batch_evaluate`

- **说明**: Compute bootstrap p-values for all factors, then apply BH.
- **内部/外部调用**: `apply_fdr`, `compute_p_value`, `items`

#### 类: `DeflatedSharpeResult`

**说明**: Result of Deflated Sharpe Ratio test.

#### 类: `DeflatedSharpeCalculator`

**说明**: Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

##### 方法: `DeflatedSharpeCalculator.__init__`


##### 方法: `DeflatedSharpeCalculator.compute`

- **说明**: Compute the Deflated Sharpe Ratio for a factor's L/S returns.
- **内部/外部调用**: `_expected_max_sr`, `mean`, `DeflatedSharpeResult`, `kurtosis`, `sqrt`, `std`, `cdf`, `skew`, `isnan`

##### 方法: `DeflatedSharpeCalculator._expected_max_sr`

- **说明**: E[max(SR)] approximation from Bailey & López de Prado (2014).
- **内部/外部调用**: `log`, `sqrt`

#### 函数: `check_significance`

- **说明**: Run all significance checks on a single factor.
- **内部/外部调用**: `compute_p_value`, `BootstrapICTester`, `compute`, `DeflatedSharpeCalculator`, `SignificanceConfig`, `compute_ci`

### 文件: `factorminer/evaluation/transaction_costs.py`

**模块说明**: Transaction cost models for realistic P&L computation.

Implements the Almgren-Chriss (2001) market impact framework, bid-ask
slippage, commissions, and A-share specific taxes. All costs are expressed
in basis points (bps) unless explicitly noted.

References
----------
Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions.
    Journal of Risk, 3(2), 5-39.
Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management.
    Academic Press.

#### 类: `TradingCosts`

**说明**: Aggregated transaction costs for a single rebalance event.

#### 类: `MarketImpactModel`

**说明**: Almgren-Chriss (2001) market impact model.

##### 方法: `MarketImpactModel.__init__`

- **内部/外部调用**:

##### 方法: `MarketImpactModel.compute_impact`

- **说明**: Compute total Almgren-Chriss impact for a batch of trades.
- **内部/外部调用**: `power`, `where`, `abs`, `sign`, `asarray`

#### 类: `SlippageModel`

**说明**: Bid-ask spread slippage model.

##### 方法: `SlippageModel.__init__`

- **说明**: Parameters
- **内部/外部调用**:

##### 方法: `SlippageModel.compute_slippage`

- **说明**: Compute one-way slippage for a set of trades.
- **内部/外部调用**: `where`, `full`, `asarray`

#### 类: `TransactionCostCalculator`

**说明**: Aggregate all transaction cost components for a portfolio rebalance.

##### 方法: `TransactionCostCalculator.__init__`

- **内部/外部调用**: `SlippageModel`, `MarketImpactModel`

##### 方法: `TransactionCostCalculator.compute_total_cost`

- **说明**: Compute all-in transaction costs for a single rebalance event.
- **内部/外部调用**: `compute_slippage`, `max`, `compute_impact`, `abs`, `TradingCosts`, `sign`, `sum`, `asarray`

##### 方法: `TransactionCostCalculator.for_ashare`

- **说明**: Convenience constructor with A-share defaults.
- **内部/外部调用**: `cls`, `SlippageModel`, `MarketImpactModel`

##### 方法: `TransactionCostCalculator.for_crypto`

- **说明**: Convenience constructor with crypto exchange defaults.
- **内部/外部调用**: `cls`, `SlippageModel`, `MarketImpactModel`

### 文件: `factorminer/memory/__init__.py`

**模块说明**: Experience memory system for mining loop feedback.

Implements the memory M = {S, P_succ, P_fail, I} with operators:
- F(M, tau): Memory Formation - extract experience from mining trajectory
- E(M, M_form): Memory Evolution - consolidate and prune memory
- R(M, L): Memory Retrieval - context-dependent retrieval for LLM prompts

Phase 2 additions:
- Knowledge Graph: factor lineage and structural analysis
- Embeddings: semantic formula similarity and deduplication
- Enhanced Retrieval: KG + embedding augmented retrieval

### 文件: `factorminer/memory/embeddings.py`

**模块说明**: Semantic formula embeddings for factor similarity and deduplication.

Converts DSL formulas into natural language descriptions and encodes
them as dense vectors. Supports:
- sentence-transformers for high-quality embeddings (optional)
- FAISS for fast k-NN search (optional)
- TF-IDF fallback when sentence-transformers is unavailable
- Brute-force cosine fallback when FAISS is unavailable

#### 类: `FormulaEmbedder`

**说明**: Embed DSL formulas as dense vectors for similarity search.

##### 方法: `FormulaEmbedder.__init__`


##### 方法: `FormulaEmbedder.embed`

- **说明**: Compute (or retrieve cached) embedding for a formula.
- **内部/外部调用**: `_formula_to_text`, `_encode`, `append`

##### 方法: `FormulaEmbedder.remove`

- **说明**: Remove a cached embedding by factor id.
- **内部/外部调用**: `pop`

##### 方法: `FormulaEmbedder.clear`

- **说明**: Clear all cached embeddings and search state.
- **内部/外部调用**: `clear`

##### 方法: `FormulaEmbedder.cache_size`

- **说明**: Return the number of cached factor embeddings.
- **内部/外部调用**:

##### 方法: `FormulaEmbedder.find_nearest`

- **说明**: Find the *k* most similar cached formulas.
- **内部/外部调用**: `_brute_force_search`, `_formula_to_text`, `_encode`, `min`, `_faiss_search`

##### 方法: `FormulaEmbedder.is_semantic_duplicate`

- **说明**: Check if *formula* is a near-duplicate of a cached factor.
- **内部/外部调用**: `find_nearest`

##### 方法: `FormulaEmbedder._formula_to_text`

- **说明**: Convert a DSL formula into a natural-language description.
- **内部/外部调用**: `replace`, `sub`, `lower`, `sorted`, `strip`

##### 方法: `FormulaEmbedder._encode`

- **说明**: Encode a single text string into a unit-norm vector.
- **内部/外部调用**: `_encode_hash`, `_encode_transformer`, `_encode_tfidf`

##### 方法: `FormulaEmbedder._encode_transformer`

- **内部/外部调用**: `flatten`, `norm`, `SentenceTransformer`, `encode`, `asarray`

##### 方法: `FormulaEmbedder._encode_tfidf`

- **说明**: Encode using TF-IDF over all cached texts + query.
- **内部/外部调用**: `flatten`, `norm`, `values`, `fit_transform`, `append`, `TfidfVectorizer`, `asarray`, `toarray`, `enumerate`

##### 方法: `FormulaEmbedder._encode_hash`

- **说明**: Ultra-simple hash-based embedding fallback.
- **内部/外部调用**: `zeros`, `split`, `hash`, `norm`

##### 方法: `FormulaEmbedder._rebuild_index`

- **说明**: Rebuild the FAISS ``IndexFlatIP`` from cached embeddings.
- **内部/外部调用**: `add`, `stack`, `IndexFlatIP`

##### 方法: `FormulaEmbedder._faiss_search`

- **内部/外部调用**: `_rebuild_index`, `_brute_force_search`, `reshape`, `append`, `search`, `zip`

##### 方法: `FormulaEmbedder._brute_force_search`

- **内部/外部调用**: `sort`, `dot`, `append`

### 文件: `factorminer/memory/evolution.py`

**模块说明**: Memory Evolution operator E(M, M_form).

Consolidates newly formed experience into the existing memory:
- Merges redundant success/failure patterns
- Discards low-utility entries
- Reclassifies patterns that have changed behavior
- Caps memory size according to configuration limits

#### 函数: `_merge_success_patterns`

- **说明**: Merge new success patterns into existing ones.
- **内部/外部调用**: `values`, `add`, `append`, `SuccessPattern`

#### 函数: `_merge_forbidden_directions`

- **说明**: Merge new forbidden directions into existing ones.
- **内部/外部调用**: `values`, `add`, `append`, `ForbiddenDirection`

#### 函数: `_merge_insights`

- **说明**: Merge new insights into existing, deduplicating similar ones.
- **内部/外部调用**: `max`, `enumerate`, `lower`, `split`, `append`

#### 函数: `_reclassify_patterns`

- **说明**: Reclassify patterns that have changed behavior.
- **内部/外部调用**: `_names_overlap`, `append`, `ForbiddenDirection`

#### 函数: `_names_overlap`

- **说明**: Check if two pattern names refer to the same concept.
- **内部/外部调用**: `replace`, `lower`, `split`, `min`

#### 函数: `_prune_low_utility`

- **说明**: Remove entries with too few occurrences to be reliable.

#### 函数: `_cap_memory_size`

- **说明**: Enforce maximum memory sizes by keeping the most useful entries.
- **内部/外部调用**: `sorted`

#### 函数: `evolve_memory`

- **说明**: Memory Evolution operator E(M, M_form).
- **内部/外部调用**: `_prune_low_utility`, `ExperienceMemory`, `_merge_success_patterns`, `_merge_forbidden_directions`, `_reclassify_patterns`, `_merge_insights`, `_cap_memory_size`

#### 函数: `apply_confidence_decay`

- **说明**: Return new ExperienceMemory with decayed pattern confidences.
- **内部/外部调用**: `replace`, `append`

#### 函数: `bump_pattern_confidence`

- **说明**: Return new ExperienceMemory with confidence boosted for matching patterns.
- **内部/外部调用**: `replace`, `lower`, `append`, `any`, `min`

#### 函数: `penalise_pattern_confidence`

- **说明**: Return new ExperienceMemory with confidence penalised for matching patterns.
- **内部/外部调用**: `replace`, `lower`, `append`, `any`

### 文件: `factorminer/memory/experience_memory.py`

**模块说明**: Main ExperienceMemory manager class.

Provides the high-level API for the experience memory system:
- Initializes with default patterns from the paper (Tables 4 and 5)
- Persists to/from JSON
- update(trajectory) orchestrates formation + evolution
- retrieve(library_state) performs context-dependent retrieval
- Optional knowledge graph and embedding support for Phase 2

#### 类: `ExperienceMemoryManager`

**说明**: High-level manager for the experience memory system.

##### 方法: `ExperienceMemoryManager.__init__`

- **内部/外部调用**: `MiningState`, `warning`, `ExperienceMemory`, `_default_forbidden_directions`, `FormulaEmbedder`, `FactorKnowledgeGraph`, `_default_success_patterns`, `_default_insights`

##### 方法: `ExperienceMemoryManager.version`


##### 方法: `ExperienceMemoryManager.update`

- **说明**: Process a batch trajectory: formation + evolution.
- **内部/外部调用**: `form_memory`, `_update_knowledge_graph`, `evolve_memory`, `sum`, `get`

##### 方法: `ExperienceMemoryManager.retrieve`

- **说明**: Retrieve context-dependent memory signal for LLM prompt.
- **内部/外部调用**: `retrieve_memory`, `retrieve_memory_enhanced`

##### 方法: `ExperienceMemoryManager.save`

- **说明**: Persist memory to a JSON file.
- **内部/外部调用**: `dump`, `with_name`, `save`, `open`, `to_dict`, `mkdir`, `Path`

##### 方法: `ExperienceMemoryManager.load`

- **说明**: Load memory from a JSON file.
- **内部/外部调用**: `with_name`, `FormulaEmbedder`, `open`, `from_dict`, `exists`, `get`, `Path`, `load`

##### 方法: `ExperienceMemoryManager.get_stats`

- **说明**: Return summary statistics about the current memory state.
- **内部/外部调用**: `find_saturated_regions`, `get_factor_count`, `sorted`, `round`, `sum`, `get`, `get_edge_count`

##### 方法: `ExperienceMemoryManager.reset`

- **说明**: Reset memory to initial state with default knowledge base.
- **内部/外部调用**: `MiningState`, `ExperienceMemory`, `_default_forbidden_directions`, `FormulaEmbedder`, `FactorKnowledgeGraph`, `_default_success_patterns`, `_default_insights`

##### 方法: `ExperienceMemoryManager._update_knowledge_graph`

- **说明**: Add factors from a trajectory to the knowledge graph.
- **内部/外部调用**: `compile`, `FactorNode`, `add_correlation_edge`, `add_factor`, `embed`, `findall`, `append`, `get`

#### 函数: `_default_success_patterns`

- **说明**: Initial success patterns from FactorMiner Table 4.
- **内部/外部调用**: `SuccessPattern`

#### 函数: `_default_forbidden_directions`

- **说明**: Initial forbidden directions from FactorMiner Table 5.
- **内部/外部调用**: `ForbiddenDirection`

#### 函数: `_default_insights`

- **说明**: Initial strategic insights from the paper.
- **内部/外部调用**: `StrategicInsight`

### 文件: `factorminer/memory/formation.py`

**模块说明**: Memory Formation operator F(M, tau).

Analyzes a mining trajectory tau (batch of evaluated candidates with IC,
correlation, admission results) and extracts new experience:
- Successful patterns from admitted factors
- Forbidden directions from high-correlation rejections
- Strategic insights about what works across the batch

#### 函数: `_extract_operators`

- **说明**: Extract operator names from a DSL formula string.
- **内部/外部调用**: `findall`

#### 函数: `_extract_features`

- **说明**: Extract feature references from a DSL formula string.
- **内部/外部调用**: `findall`

#### 函数: `_matches_pattern`

- **说明**: Check if a formula matches a pattern based on keyword presence.
- **内部/外部调用**: `_extract_operators`, `_extract_features`, `upper`, `min`, `any`, `sum`

#### 函数: `_classify_success_pattern`

- **说明**: Try to classify a formula into a known success pattern category.
- **内部/外部调用**: `_matches_pattern`, `items`

#### 函数: `_classify_forbidden_direction`

- **说明**: Try to classify a formula into a known forbidden direction.
- **内部/外部调用**: `_matches_pattern`, `items`

#### 函数: `_analyze_admissions`

- **说明**: Split trajectory into admitted and rejected candidates.
- **内部/外部调用**: `get`, `append`

#### 函数: `_extract_success_patterns`

- **说明**: Extract new or reinforced success patterns from admitted factors.
- **内部/外部调用**: `values`, `join`, `_extract_operators`, `append`, `SuccessPattern`, `_classify_success_pattern`, `get`

#### 函数: `_extract_forbidden_directions`

- **说明**: Extract new or reinforced forbidden directions from rejections.
- **内部/外部调用**: `values`, `lower`, `join`, `_extract_operators`, `_extract_features`, `_classify_forbidden_direction`, `append`, `ForbiddenDirection`, `get`

#### 函数: `_derive_insights`

- **说明**: Derive higher-level strategic insights from a batch.
- **内部/外部调用**: `max`, `_extract_operators`, `_extract_features`, `StrategicInsight`, `append`, `most_common`, `Counter`, `any`, `sum`, `get`

#### 函数: `form_memory`

- **说明**: Memory Formation operator F(M, tau).
- **内部/外部调用**: `MiningState`, `max`, `ExperienceMemory`, `_extract_success_patterns`, `_extract_forbidden_directions`, `_compute_domain_saturation`, `_analyze_admissions`, `get`, `_derive_insights`

#### 函数: `_compute_domain_saturation`

- **说明**: Compute per-category domain saturation metrics.
- **内部/外部调用**: `defaultdict`, `items`, `_classify_success_pattern`, `get`

### 文件: `factorminer/memory/kg_retrieval.py`

**模块说明**: Enhanced memory retrieval combining Knowledge Graph + Embeddings + flat memory.

#### 函数: `retrieve_memory_enhanced`

- **说明**: Enhanced memory retrieval operator R+(M, L, KG, E).
- **内部/外部调用**: `add`, `find_saturated_regions`, `_find_semantic_gaps`, `get_operator_cooccurrence`, `join`, `_collect_semantic_context`, `sorted`, `find_complementary_patterns`, `_describe_factor_node`, `append`, `_seed_embedder_from_memory`, `items`, `_describe_conflict_cluster`, `retrieve_memory`, `get`

#### 函数: `_find_semantic_gaps`

- **说明**: Identify success-pattern operators with poor semantic coverage.
- **内部/外部调用**: `add`, `compile`, `sorted`, `list_factor_nodes`, `update`, `finditer`, `find_nearest`, `group`

#### 函数: `_seed_embedder_from_memory`

- **说明**: Ensure the embedder cache reflects the current known factors.
- **内部/外部调用**: `add`, `embed`, `list_factor_nodes`, `get`

#### 函数: `_collect_semantic_context`

- **说明**: Collect semantically similar neighbors and duplicate warnings.
- **内部/外部调用**: `add`, `_describe_factor_node`, `append`, `min`, `find_nearest`, `get`

#### 函数: `_describe_factor_node`

- **说明**: Render a factor node into short prompt-friendly text.
- **内部/外部调用**: `get_factor_node`

#### 函数: `_describe_conflict_cluster`

- **说明**: Render one saturated cluster into short text.
- **内部/外部调用**: `join`, `sorted`, `_describe_factor_node`

### 文件: `factorminer/memory/knowledge_graph.py`

**模块说明**: Factor Knowledge Graph for lineage tracking and structural analysis.

Uses a NetworkX DiGraph to model relationships between factors, operators,
and feature inputs. Supports:
- Factor derivation lineage (parent -> child mutations)
- Correlation-based edges for saturation detection
- Operator co-occurrence analysis for diversity guidance
- Complementary pattern discovery via BFS

#### 类: `EdgeType`

**说明**: Types of edges in the factor knowledge graph.

#### 类: `FactorNode`

**说明**: A node in the factor knowledge graph representing a single factor.

##### 方法: `FactorNode.to_dict`

- **内部/外部调用**: `tolist`, `asdict`

##### 方法: `FactorNode.from_dict`

- **内部/外部调用**: `array`, `cls`, `get`

#### 类: `FactorKnowledgeGraph`

**说明**: Directed graph tracking factor lineage and relationships.

##### 方法: `FactorKnowledgeGraph.__init__`

- **内部/外部调用**: `_ensure_networkx`, `DiGraph`

##### 方法: `FactorKnowledgeGraph.add_factor`

- **说明**: Add or replace a factor node and auto-create USES_OPERATOR edges.
- **内部/外部调用**: `to_dict`, `remove_factor`, `add_node`, `add_edge`, `has_node`

##### 方法: `FactorKnowledgeGraph.get_factor_node`

- **说明**: Return a factor node by id, or ``None`` if missing.
- **内部/外部调用**: `get`, `from_dict`

##### 方法: `FactorKnowledgeGraph.iter_factor_nodes`

- **说明**: Yield factor nodes currently present in the graph.
- **内部/外部调用**: `get`, `nodes`, `from_dict`

##### 方法: `FactorKnowledgeGraph.list_factor_nodes`

- **说明**: Return all factor nodes as a list.
- **内部/外部调用**: `iter_factor_nodes`

##### 方法: `FactorKnowledgeGraph.add_correlation_edge`

- **说明**: Add a CORRELATED_WITH edge if ``|rho| >= threshold``.
- **内部/外部调用**: `add_edge`, `abs`

##### 方法: `FactorKnowledgeGraph.add_derivation_edge`

- **说明**: Add a DERIVED_FROM edge from *child* to *parent*.
- **内部/外部调用**: `add_edge`

##### 方法: `FactorKnowledgeGraph.remove_factor`

- **说明**: Remove a factor and prune orphaned auxiliary nodes.
- **内部/外部调用**: `has_node`, `remove_node`, `_prune_orphan_aux_nodes`

##### 方法: `FactorKnowledgeGraph.find_complementary_patterns`

- **说明**: Find factors complementary to *factor_id* via BFS.
- **内部/外部调用**: `add`, `issubset`, `_get_operators`, `in_edges`, `append`, `neighbors`, `to_undirected`, `edges`, `range`, `has_node`, `get`

##### 方法: `FactorKnowledgeGraph.find_saturated_regions`

- **说明**: Find clusters of highly correlated factors.
- **内部/外部调用**: `connected_components`, `abs`, `Graph`, `edges`, `add_edge`, `get`

##### 方法: `FactorKnowledgeGraph.get_operator_cooccurrence`

- **说明**: Count operator pair co-occurrences across admitted factors.
- **内部/外部调用**: `get`, `defaultdict`, `sorted`, `range`, `nodes`

##### 方法: `FactorKnowledgeGraph.get_factor_count`

- **说明**: Return the number of factor nodes in the graph.
- **内部/外部调用**: `get`, `sum`, `nodes`

##### 方法: `FactorKnowledgeGraph.get_edge_count`

- **说明**: Return total number of edges in the graph.
- **内部/外部调用**: `number_of_edges`

##### 方法: `FactorKnowledgeGraph.to_dict`

- **说明**: Serialize to a JSON-compatible dict via ``nx.node_link_data``.
- **内部/外部调用**: `node_link_data`

##### 方法: `FactorKnowledgeGraph.from_dict`

- **说明**: Deserialize from a dict produced by :meth:`to_dict`.
- **内部/外部调用**: `node_link_graph`, `cls`

##### 方法: `FactorKnowledgeGraph.save`

- **说明**: Persist the graph to a JSON file.
- **内部/外部调用**: `dump`, `open`, `to_dict`, `mkdir`, `Path`

##### 方法: `FactorKnowledgeGraph.load`

- **说明**: Load a graph from a JSON file.
- **内部/外部调用**: `open`, `from_dict`, `Path`, `load`

##### 方法: `FactorKnowledgeGraph._get_operators`

- **说明**: Return the set of operator names used by a factor.
- **内部/外部调用**: `add`, `removeprefix`, `edges`, `get`

##### 方法: `FactorKnowledgeGraph._prune_orphan_aux_nodes`

- **说明**: Remove operator nodes that are no longer referenced.
- **内部/外部调用**: `get`, `nodes`, `degree`, `remove_nodes_from`

#### 函数: `_ensure_networkx`

- **说明**: Raise a clear error if networkx is not installed.
- **内部/外部调用**: `ImportError`

### 文件: `factorminer/memory/memory_store.py`

**模块说明**: Data structures for the FactorMiner experience memory system.

Implements the experience memory M = {S, P_succ, P_fail, I} where:
- S: Mining state tracking global evolution of the factor library
- P_succ: Success patterns (recommended mining directions)
- P_fail: Forbidden directions (directions to avoid)
- I: Strategic insights (high-level lessons)

#### 类: `MiningState`

**说明**: Tracks the global evolution of the factor library (S).

##### 方法: `MiningState.to_dict`

- **内部/外部调用**: `asdict`

##### 方法: `MiningState.from_dict`

- **内部/外部调用**: `cls`, `get`

#### 类: `SuccessPattern`

**说明**: A recommended mining direction (P_succ).

##### 方法: `SuccessPattern.to_dict`

- **内部/外部调用**: `asdict`

##### 方法: `SuccessPattern.from_dict`

- **内部/外部调用**: `cls`, `get`

#### 类: `ForbiddenDirection`

**说明**: A forbidden mining direction (P_fail).

##### 方法: `ForbiddenDirection.to_dict`

- **内部/外部调用**: `asdict`

##### 方法: `ForbiddenDirection.from_dict`

- **内部/外部调用**: `cls`, `get`

#### 类: `StrategicInsight`

**说明**: High-level lesson from mining (I).

##### 方法: `StrategicInsight.to_dict`

- **内部/外部调用**: `asdict`

##### 方法: `StrategicInsight.from_dict`

- **内部/外部调用**: `cls`, `get`

#### 类: `ExperienceMemory`

**说明**: The complete experience memory M = {S, P_succ, P_fail, I}.

##### 方法: `ExperienceMemory.to_dict`

- **内部/外部调用**: `to_dict`

##### 方法: `ExperienceMemory.from_dict`

- **内部/外部调用**: `get`, `cls`, `from_dict`

### 文件: `factorminer/memory/online_regime_memory.py`

**模块说明**: Online regime-aware memory system for FactorMiner.

Addresses FactorMiner's core limitation: static, offline-only memory that
ignores regime changes.  This module provides:

- ``RegimeSpecificPattern`` / ``RegimeSpecificPatternStore``
  — per-regime success/failure pattern storage with IC-based scoring

- ``OnlineMemoryUpdater``
  — streaming memory update with exponential forgetting and regime-change hooks

- ``RegimeTransitionForecaster``
  — logistic-regression-based next-regime predictor for proactive memory prep

- ``OnlineRegimeMemory``
  — top-level orchestrator integrating all components

- ``MemoryForgetCurve``
  — snapshot tracker for visualising and analysing memory decay

All components are:
  * Thread-safe (``threading.RLock``)
  * Serialisable (``to_dict`` / ``from_dict`` + ``pickle`` compatible)
  * Streaming-fast (< 1 ms per ``update`` call with normal loads)
  * Pure Python + NumPy + scikit-learn (no additional dependencies)

#### 类: `MemorySignal`

**说明**: Structured memory signal for LLM prompt injection.

##### 方法: `MemorySignal.to_dict`

- **内部/外部调用**: `to_dict`

#### 类: `RegimeSpecificPattern`

**说明**: A formula pattern with per-regime performance statistics.

##### 方法: `RegimeSpecificPattern.update_ic`

- **说明**: Online update of IC statistics using an EW running mean.
- **内部/外部调用**: `min`, `abs`

##### 方法: `RegimeSpecificPattern.to_dict`

- **内部/外部调用**: `isoformat`, `round`, `to_dict`

##### 方法: `RegimeSpecificPattern.from_dict`

- **内部/外部调用**: `cls`, `isoformat`, `replace`, `fromisoformat`, `from_dict`, `now`, `get`

#### 类: `RegimeSpecificPatternStore`

**说明**: Thread-safe store for regime-specific formula patterns.

##### 方法: `RegimeSpecificPatternStore.__init__`

- **内部/外部调用**: `RLock`

##### 方法: `RegimeSpecificPatternStore.add_pattern`

- **说明**: Add or update a pattern observation.
- **内部/外部调用**: `RegimeSpecificPattern`, `_evict_weakest`, `abs`, `min`, `update_ic`, `items`

##### 方法: `RegimeSpecificPatternStore.retrieve_for_regime`

- **说明**: Retrieve patterns most relevant to the current regime.
- **内部/外部调用**: `values`, `abs`, `similarity`, `append`, `sort`

##### 方法: `RegimeSpecificPatternStore.get_cross_regime_patterns`

- **说明**: Return patterns that generalise well across regimes.
- **内部/外部调用**: `values`, `sort`, `abs`, `append`

##### 方法: `RegimeSpecificPatternStore.apply_decay`

- **说明**: Multiply all pattern confidences by ``decay_factor`` and prune weak ones.
- **内部/外部调用**: `max`, `items`, `append`

##### 方法: `RegimeSpecificPatternStore.boost_regime_patterns`

- **说明**: Increase confidence of patterns tagged for ``regime``.
- **内部/外部调用**: `values`, `min`

##### 方法: `RegimeSpecificPatternStore.penalise_regime_patterns`

- **说明**: Decrease confidence of patterns tagged for ``regime``.
- **内部/外部调用**: `values`, `max`

##### 方法: `RegimeSpecificPatternStore.get_stats`

- **说明**: Return aggregate statistics.
- **内部/外部调用**: `values`, `mean`, `abs`, `get_cross_regime_patterns`

##### 方法: `RegimeSpecificPatternStore.to_dict`

- **内部/外部调用**: `values`, `to_dict`

##### 方法: `RegimeSpecificPatternStore.from_dict`

- **内部/外部调用**: `cls`, `get`, `from_dict`

##### 方法: `RegimeSpecificPatternStore._evict_weakest`

- **说明**: Remove the single weakest (lowest confidence * ic) pattern.
- **内部/外部调用**: `min`, `abs`

#### 类: `OnlineMemoryUpdater`

**说明**: Streaming experience-memory updater with exponential forgetting.

##### 方法: `OnlineMemoryUpdater.__init__`

- **内部/外部调用**: `RLock`, `deque`, `defaultdict`

##### 方法: `OnlineMemoryUpdater.base_memory`

- **说明**: Thread-safe read of the current base memory snapshot.

##### 方法: `OnlineMemoryUpdater.on_factor_evaluated`

- **说明**: Called immediately after each factor evaluation.
- **内部/外部调用**: `debug`, `abs`, `bump_pattern_confidence`, `perf_counter`, `append`, `_formula_matches_template`

##### 方法: `OnlineMemoryUpdater.apply_forgetting`

- **说明**: Exponentially decay pattern confidence and prune stale entries.
- **内部/外部调用**: `apply_confidence_decay`

##### 方法: `OnlineMemoryUpdater.on_regime_change`

- **说明**: React to a detected regime transition.
- **内部/外部调用**: `penalise_pattern_confidence`, `lower`, `StrategicInsight`, `append`, `any`, `bump_pattern_confidence`

##### 方法: `OnlineMemoryUpdater.get_memory_health_stats`

- **说明**: Return comprehensive health statistics for the memory system.
- **内部/外部调用**: `max`, `mean`, `round`, `sum`, `items`

##### 方法: `OnlineMemoryUpdater.to_dict`

- **内部/外部调用**: `to_dict`, `items`

##### 方法: `OnlineMemoryUpdater.from_dict`

- **内部/外部调用**: `cls`, `from_dict`, `update`, `items`, `deque`, `get`

#### 类: `RegimeTransitionForecaster`

**说明**: Logistic-regression forecaster for regime transitions.

##### 方法: `RegimeTransitionForecaster.__init__`

- **内部/外部调用**: `RLock`

##### 方法: `RegimeTransitionForecaster.record_observation`

- **说明**: Append one (features, regime) observation to the training buffer.
- **内部/外部调用**: `append`, `copy`

##### 方法: `RegimeTransitionForecaster.fit`

- **说明**: Fit (or re-fit) the logistic regression model.
- **内部/外部调用**: `warning`, `fit_transform`, `enumerate`, `LogisticRegression`, `fit`, `StandardScaler`, `array`, `items`

##### 方法: `RegimeTransitionForecaster.predict_next_regime`

- **说明**: Predict the most probable next regime.
- **内部/外部调用**: `warning`, `reshape`, `fit`, `transform`, `predict_proba`, `argmax`, `get`, `RegimeState`

##### 方法: `RegimeTransitionForecaster.prepare_memory_for_transition`

- **说明**: Pre-load (boost confidence of) patterns for the predicted regime.
- **内部/外部调用**: `boost_regime_patterns`

##### 方法: `RegimeTransitionForecaster.build_feature_vector`

- **说明**: Build a fixed-length feature vector from streaming statistics.
- **内部/外部调用**: `array`

##### 方法: `RegimeTransitionForecaster.to_dict`

- **内部/外部调用**: `tolist`, `to_dict`, `items`

##### 方法: `RegimeTransitionForecaster.from_dict`

- **内部/外部调用**: `cls`, `fit`, `from_dict`, `items`, `array`, `get`

#### 类: `OnlineRegimeMemory`

**说明**: Full online regime-aware memory system.

##### 方法: `OnlineRegimeMemory.__init__`

- **内部/外部调用**: `ExperienceMemory`, `RegimeTransitionForecaster`, `items`, `StreamingRegimeConfig`, `StreamingRegimeDetector`, `RegimeSpecificPatternStore`, `OnlineMemoryUpdater`, `RLock`, `get`, `RegimeState`

##### 方法: `OnlineRegimeMemory.update_market`

- **说明**: Process one bar of market data and update the regime state.
- **内部/外部调用**: `penalise_regime_patterns`, `boost_regime_patterns`, `predict_next_regime`, `_build_feature_vector`, `update`, `record_observation`, `on_regime_change`, `prepare_memory_for_transition`

##### 方法: `OnlineRegimeMemory.update`

- **说明**: Single update call: detect regime from market_data, update patterns.
- **内部/外部调用**: `apply_forgetting`, `on_factor_evaluated`, `abs`, `apply_decay`, `update_market`, `add_pattern`, `get`

##### 方法: `OnlineRegimeMemory.retrieve`

- **说明**: Regime-aware memory retrieval.
- **内部/外部调用**: `MemorySignal`, `predict_next_regime`, `_build_feature_vector`, `to_dict`, `_format_regime_section`, `update_market`, `retrieve_memory`, `get_cross_regime_patterns`, `get`, `retrieve_for_regime`

##### 方法: `OnlineRegimeMemory.get_full_status`

- **说明**: Comprehensive status: regime, patterns, health, forecasts.
- **内部/外部调用**: `get_memory_health_stats`, `get_stats`, `get_regime_history`, `predict_next_regime`, `_build_feature_vector`, `to_dict`, `round`, `regime_transition_probability`

##### 方法: `OnlineRegimeMemory.save`

- **说明**: Serialise to JSON.
- **内部/外部调用**: `dump`, `open`, `to_dict`, `mkdir`, `Path`

##### 方法: `OnlineRegimeMemory.load`

- **说明**: Deserialise from JSON.
- **内部/外部调用**: `open`, `_from_dict_inplace`, `load`

##### 方法: `OnlineRegimeMemory.to_dict`

- **内部/外部调用**: `to_dict`

##### 方法: `OnlineRegimeMemory.from_dict`

- **内部/外部调用**: `get`, `cls`, `from_dict`, `_from_dict_inplace`

##### 方法: `OnlineRegimeMemory._from_dict_inplace`

- **内部/外部调用**: `get`, `from_dict`

##### 方法: `OnlineRegimeMemory.__getstate__`

- **内部/外部调用**: `to_dict`

##### 方法: `OnlineRegimeMemory.__setstate__`

- **内部/外部调用**: `RLock`, `StreamingRegimeDetector`, `_from_dict_inplace`

##### 方法: `OnlineRegimeMemory._build_feature_vector`

- **说明**: Build a 12-element feature vector from the detector's EW state.
- **内部/外部调用**: `max`, `sqrt`, `clip`, `log`, `build_feature_vector`

##### 方法: `OnlineRegimeMemory._format_regime_section`

- **内部/外部调用**: `join`, `abs`, `enumerate`, `append`

#### 类: `_MemorySnapshot`

**说明**: Internal snapshot used by MemoryForgetCurve.

#### 类: `MemoryForgetCurve`

**说明**: Track and visualise how memory evolves (and forgets) over mining iterations.

##### 方法: `MemoryForgetCurve.__init__`

- **内部/外部调用**: `RLock`

##### 方法: `MemoryForgetCurve.record_snapshot`

- **说明**: Record a snapshot of the current memory state.
- **内部/外部调用**: `values`, `get_full_status`, `append`, `time`, `_MemorySnapshot`

##### 方法: `MemoryForgetCurve.get_pattern_lifetimes`

- **说明**: Estimate pattern lifetimes (iterations survived) from snapshot series.
- **内部/外部调用**: `append`, `range`

##### 方法: `MemoryForgetCurve.plot_confidence_decay`

- **说明**: Plot confidence decay and pattern count over iterations.
- **内部/外部调用**: `subplots`, `max`, `set_title`, `tight_layout`, `plot`, `show`, `grid`, `suptitle`, `set_ylabel`, `set_xlabel`

##### 方法: `MemoryForgetCurve.to_dict`


##### 方法: `MemoryForgetCurve.from_dict`

- **内部/外部调用**: `cls`, `get`, `_MemorySnapshot`, `append`

#### 函数: `_formula_matches_template`

- **说明**: Heuristic check: does a formula share structural operators with a template?
- **内部/外部调用**: `compile`, `max`, `findall`

### 文件: `factorminer/memory/retrieval.py`

**模块说明**: Memory Retrieval operator R(M, L).

Context-dependent retrieval of experience memory, producing a structured
memory signal m for injection into the LLM generation prompt.

The retrieval considers the current library state (domain saturation,
recent rejections) to select the most relevant patterns and insights.

#### 函数: `_score_success_pattern`

- **说明**: Score a success pattern for relevance given current library state.
- **内部/外部调用**: `get`, `log1p`

#### 函数: `_score_forbidden_direction`

- **说明**: Score a forbidden direction for relevance.
- **内部/外部调用**: `lower`, `split`, `log1p`, `any`

#### 函数: `_select_relevant_success`

- **说明**: Select the most relevant success patterns for the current context.
- **内部/外部调用**: `sort`, `_score_success_pattern`

#### 函数: `_select_relevant_forbidden`

- **说明**: Select the most relevant forbidden directions for the current context.
- **内部/外部调用**: `sort`, `get`, `_score_forbidden_direction`

#### 函数: `_format_library_state`

- **说明**: Format mining state as structured context for LLM prompt.
- **内部/外部调用**: `items`, `round`, `sum`, `get`

#### 函数: `_format_for_prompt`

- **说明**: Format the memory signal as structured text for LLM injection.
- **内部/外部调用**: `get`, `join`, `append`, `items`, `enumerate`

#### 函数: `retrieve_memory`

- **说明**: Memory Retrieval operator R(M, L).
- **内部/外部调用**: `_select_relevant_forbidden`, `MiningState`, `_format_for_prompt`, `_select_relevant_success`, `sorted`, `to_dict`, `_format_library_state`, `get`

### 文件: `factorminer/operators/__init__.py`

**模块说明**: Financial operators for factor expression evaluation.

Exports the central registry and all operator category modules.

### 文件: `factorminer/operators/arithmetic.py`

**模块说明**: Element-wise arithmetic operators (unary and binary).

Every function accepts arrays of shape ``(M, T)`` and returns the same shape.
Both NumPy and PyTorch implementations are provided.

#### 函数: `_eps`


#### 函数: `add_np`

- **内部/外部调用**: `add`

#### 函数: `sub_np`

- **内部/外部调用**: `subtract`

#### 函数: `mul_np`

- **内部/外部调用**: `multiply`

#### 函数: `div_np`

- **内部/外部调用**: `full_like`, `abs`

#### 函数: `neg_np`

- **内部/外部调用**: `negative`

#### 函数: `abs_np`

- **内部/外部调用**: `abs`

#### 函数: `sign_np`

- **内部/外部调用**: `sign`

#### 函数: `log_np`

- **说明**: log(1 + |x|) * sign(x) -- safe log that handles negatives.
- **内部/外部调用**: `sign`, `abs`, `log1p`

#### 函数: `sqrt_np`

- **说明**: sqrt(|x|) * sign(x).
- **内部/外部调用**: `sign`, `abs`, `sqrt`

#### 函数: `square_np`

- **内部/外部调用**: `square`

#### 函数: `inv_np`

- **内部/外部调用**: `full_like`, `abs`

#### 函数: `pow_np`

- **说明**: x^y with safe handling.
- **内部/外部调用**: `power`, `where`, `abs`, `sign`, `errstate`, `isnan`

#### 函数: `max_np`

- **内部/外部调用**: `fmax`

#### 函数: `min_np`

- **内部/外部调用**: `fmin`

#### 函数: `clip_np`

- **内部/外部调用**: `clip`

#### 函数: `exp_np`

- **说明**: Clamped exp to avoid overflow.
- **内部/外部调用**: `clip`, `exp`

#### 函数: `tanh_np`

- **内部/外部调用**: `tanh`

#### 函数: `signed_power_np`

- **内部/外部调用**: `sign`, `abs`, `power`

#### 函数: `power_np`

- **内部/外部调用**: `errstate`, `power`

#### 函数: `add_torch`


#### 函数: `sub_torch`


#### 函数: `mul_torch`


#### 函数: `div_torch`

- **内部/外部调用**: `full_like`, `abs`

#### 函数: `neg_torch`


#### 函数: `abs_torch`

- **内部/外部调用**: `abs`

#### 函数: `sign_torch`

- **内部/外部调用**: `sign`

#### 函数: `log_torch`

- **内部/外部调用**: `sign`, `abs`, `log1p`

#### 函数: `sqrt_torch`

- **内部/外部调用**: `sign`, `abs`, `sqrt`

#### 函数: `square_torch`


#### 函数: `inv_torch`

- **内部/外部调用**: `full_like`, `abs`

#### 函数: `pow_torch`

- **内部/外部调用**: `where`, `abs`, `sign`, `pow`, `tensor`, `isnan`

#### 函数: `max_torch`

- **内部/外部调用**: `fmax`

#### 函数: `min_torch`

- **内部/外部调用**: `fmin`

#### 函数: `clip_torch`

- **内部/外部调用**: `clamp`

#### 函数: `exp_torch`

- **内部/外部调用**: `clamp`, `exp`

#### 函数: `tanh_torch`

- **内部/外部调用**: `tanh`

#### 函数: `signed_power_torch`

- **内部/外部调用**: `sign`, `abs`, `pow`

#### 函数: `power_torch`

- **内部/外部调用**: `pow`

### 文件: `factorminer/operators/auto_inventor.py`

**模块说明**: Automated operator invention via LLM-guided proposal and validation.

Uses an LLM to propose novel operator definitions (as NumPy functions),
validates them in a sandboxed environment, and checks for differentiation
from existing operators and IC contribution.

#### 类: `ProposedOperator`

**说明**: A single operator proposal generated by the LLM.

#### 类: `ValidationResult`

**说明**: Result of validating a proposed operator.

#### 类: `OperatorInventor`

**说明**: Proposes and validates new operators using an LLM.

##### 方法: `OperatorInventor.__init__`

- **内部/外部调用**:

##### 方法: `OperatorInventor.propose_operators`

- **说明**: Ask the LLM to propose new operators.
- **内部/外部调用**: `_parse_proposals`, `_build_proposal_prompt`, `generate`, `info`

##### 方法: `OperatorInventor.validate_operator`

- **说明**: Validate a proposed operator through compilation, execution, and IC check.
- **内部/外部调用**: `fn`, `sum`, `_check_differentiation`, `_compile_safely`, `_measure_ic_contribution`, `ValidationResult`, `isnan`

##### 方法: `OperatorInventor._build_proposal_prompt`

- **说明**: Format the user prompt for operator proposals.
- **内部/外部调用**: `join`, `sorted`, `items`, `append`

##### 方法: `OperatorInventor._parse_proposals`

- **说明**: Parse LLM output into ProposedOperator objects.
- **内部/外部调用**: `compile`, `replace`, `findall`, `debug`, `append`, `items`, `ProposedOperator`, `get`, `loads`

##### 方法: `OperatorInventor._compile_safely`

- **说明**: Compile operator code in a restricted sandbox.
- **内部/外部调用**: `callable`, `warning`, `lower`, `exec`, `get`

##### 方法: `OperatorInventor._check_differentiation`

- **说明**: Check that the operator output is not too correlated with existing operators.
- **内部/外部调用**: `fn`, `flatten`, `get`, `np_fn`, `corrcoef`, `abs`, `keys`, `sum`, `info`, `isnan`

##### 方法: `OperatorInventor._measure_ic_contribution`

- **说明**: Measure the Information Coefficient of a simple factor using the operator.
- **内部/外部调用**: `fn`, `isnan`, `mean`, `corrcoef`, `append`, `full_like`, `range`, `sum`, `argsort`

### 文件: `factorminer/operators/c_backend.py`

**模块说明**: Compiled CPU backend using Bottleneck with NumPy fallbacks.

#### 函数: `backend_available`

- **说明**: Return whether the compiled Bottleneck backend is importable.

#### 函数: `_full_window`

- **内部/外部调用**: `copy`, `asarray`

#### 函数: `mean_c`

- **内部/外部调用**: `mean_np`, `_full_window`, `move_mean`

#### 函数: `std_c`

- **内部/外部调用**: `move_std`, `std_np`, `_full_window`

#### 函数: `sum_c`

- **内部/外部调用**: `sum_np`, `move_sum`, `_full_window`

#### 函数: `ts_max_c`

- **内部/外部调用**: `ts_max_np`, `_full_window`, `move_max`

#### 函数: `ts_min_c`

- **内部/外部调用**: `move_min`, `ts_min_np`, `_full_window`

#### 函数: `ts_argmax_c`

- **内部/外部调用**: `move_argmax`, `_full_window`, `ts_argmax_np`, `asarray`

#### 函数: `ts_argmin_c`

- **内部/外部调用**: `move_argmin`, `ts_argmin_np`, `_full_window`, `asarray`

#### 函数: `sma_c`

- **内部/外部调用**: `sma_np`, `mean_c`

#### 函数: `delta_c`

- **内部/外部调用**: `delta_np`

#### 函数: `corr_c`

- **内部/外部调用**: `corr_np`

#### 函数: `cov_c`

- **内部/外部调用**: `cov_np`

#### 函数: `beta_c`

- **内部/外部调用**: `beta_np`

#### 函数: `resid_c`

- **内部/外部调用**: `resid_np`

#### 函数: `wma_c`

- **内部/外部调用**: `wma_np`

#### 函数: `ts_rank_c`

- **内部/外部调用**: `ts_rank_np`

### 文件: `factorminer/operators/crosssectional.py`

**模块说明**: Cross-sectional operators (across M assets at each time step t).

Input shape: ``(M, T)`` -> output shape ``(M, T)``.
Operations are performed along axis=0 (the asset dimension) for every column.

#### 函数: `cs_rank_np`

- **说明**: Cross-sectional percentile rank -- key GPU target (26x speedup).
- **内部/外部调用**: `astype`, `full_like`, `range`, `sum`, `argsort`, `isnan`

#### 函数: `cs_zscore_np`

- **说明**: Cross-sectional z-score.
- **内部/外部调用**: `nanstd`, `where`, `nanmean`, `errstate`

#### 函数: `cs_demean_np`

- **说明**: Subtract cross-sectional mean.
- **内部/外部调用**: `nanmean`

#### 函数: `cs_scale_np`

- **说明**: Scale to unit L1 norm cross-sectionally.
- **内部/外部调用**: `nansum`, `where`, `abs`, `errstate`

#### 函数: `cs_neutralize_np`

- **说明**: Industry-neutralize (simplified: demean).
- **内部/外部调用**: `cs_demean_np`

#### 函数: `cs_quantile_np`

- **说明**: Assign each asset to a quantile bin (0 .. n_bins-1) cross-sectionally.
- **内部/外部调用**: `floor`, `clip`, `astype`, `full_like`, `range`, `sum`, `argsort`, `isnan`

#### 函数: `cs_rank_torch`

- **说明**: Cross-sectional percentile rank -- fully vectorized for GPU.
- **内部/外部调用**: `isnan`, `clamp`, `sum`, `argsort`, `clone`

#### 函数: `cs_zscore_torch`

- **内部/外部调用**: `nan_to_num`, `nanmean`, `where`, `clamp`, `sqrt`, `pow`, `sum`, `tensor`, `isnan`

#### 函数: `cs_demean_torch`

- **内部/外部调用**: `nanmean`

#### 函数: `cs_scale_torch`

- **内部/外部调用**: `nansum`, `where`, `abs`, `tensor`

#### 函数: `cs_neutralize_torch`

- **内部/外部调用**: `cs_demean_torch`

#### 函数: `cs_quantile_torch`

- **内部/外部调用**: `floor`, `clamp`, `sum`, `argsort`, `isnan`, `clone`

### 文件: `factorminer/operators/custom.py`

**模块说明**: Custom operator storage, registration, and persistence.

Manages operators invented by the auto-inventor: registers them into the
global operator registry at runtime, and persists them to disk as JSON
metadata plus Python source files for reload across sessions.

#### 类: `CustomOperator`

**说明**: A validated, auto-invented operator ready for registration.

#### 类: `CustomOperatorStore`

**说明**: Manages custom operator lifecycle: register, persist, and reload.

##### 方法: `CustomOperatorStore.__init__`

- **内部/外部调用**: `Path`

##### 方法: `CustomOperatorStore.register`

- **说明**: Register a custom operator into both global registries.
- **内部/外部调用**: `info`

##### 方法: `CustomOperatorStore.save`

- **说明**: Persist all custom operators to disk.
- **内部/外部调用**: `dumps`, `append`, `mkdir`, `info`, `write_text`, `items`

##### 方法: `CustomOperatorStore.load`

- **说明**: Load custom operators from disk, recompile, and re-register.
- **内部/外部调用**: `warning`, `register`, `debug`, `open`, `OperatorSpec`, `items`, `CustomOperator`, `exists`, `_compile_operator_code`, `info`, `read_text`, `get`, `load`

##### 方法: `CustomOperatorStore.list_operators`

- **说明**: Return names of all registered custom operators.
- **内部/外部调用**: `sorted`, `keys`

##### 方法: `CustomOperatorStore.get_operator`

- **说明**: Look up a custom operator by name.
- **内部/外部调用**: `get`

#### 函数: `_compile_operator_code`

- **说明**: Compile operator code in a restricted sandbox.
- **内部/外部调用**: `callable`, `warning`, `exec`, `get`

### 文件: `factorminer/operators/gpu_backend.py`

**模块说明**: GPU acceleration utilities for FactorMiner operators.

Provides device management, tensor conversion helpers, and batch execution
for parallel factor evaluation on CUDA GPUs with automatic CPU fallback.

#### 类: `DeviceManager`

**说明**: Singleton-style helper that picks the best available device.

##### 方法: `DeviceManager.__init__`


##### 方法: `DeviceManager.device`

- **内部/外部调用**: `device`

##### 方法: `DeviceManager._select_device`

- **内部/外部调用**: `is_available`, `device`

##### 方法: `DeviceManager.is_gpu`


##### 方法: `DeviceManager.reset`


#### 函数: `to_tensor`

- **说明**: Convert a NumPy array to a PyTorch tensor on the target device.
- **内部/外部调用**: `device`, `as_tensor`, `ascontiguousarray`, `to`

#### 函数: `to_numpy`

- **说明**: Convert a PyTorch tensor back to a NumPy array.
- **内部/外部调用**: `numpy`, `detach`, `cpu`

#### 函数: `batch_execute`

- **说明**: Execute a function over multiple parameter sets.
- **内部/外部调用**: `fn`, `append`

#### 函数: `torch_available`

- **说明**: Return True if PyTorch is importable.

### 文件: `factorminer/operators/logical.py`

**模块说明**: Conditional and comparison operators (element-wise).

All operators are element-wise on ``(M, T)`` arrays.
Boolean-like outputs use ``1.0`` / ``0.0`` (float), not Python bool.

#### 函数: `if_else_np`

- **说明**: Where cond > 0 return x, else y.  NaN in cond -> NaN.
- **内部/外部调用**: `where`, `isnan`

#### 函数: `greater_np`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `less_np`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `greater_equal_np`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `less_equal_np`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `equal_np`

- **内部/外部调用**: `where`, `abs`, `isnan`

#### 函数: `and_np`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `or_np`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `not_np`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `sign_np`

- **内部/外部调用**: `sign`

#### 函数: `max2_np`

- **内部/外部调用**: `fmax`

#### 函数: `min2_np`

- **内部/外部调用**: `fmin`

#### 函数: `ne_np`

- **内部/外部调用**: `where`, `abs`, `isnan`

#### 函数: `if_else_torch`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `greater_torch`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `less_torch`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `greater_equal_torch`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `less_equal_torch`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `equal_torch`

- **内部/外部调用**: `where`, `abs`, `isnan`

#### 函数: `and_torch`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `or_torch`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `not_torch`

- **内部/外部调用**: `where`, `isnan`

#### 函数: `sign_torch`

- **内部/外部调用**: `sign`

#### 函数: `max2_torch`

- **内部/外部调用**: `fmax`

#### 函数: `min2_torch`

- **内部/外部调用**: `fmin`

#### 函数: `ne_torch`

- **内部/外部调用**: `where`, `abs`, `isnan`

### 文件: `factorminer/operators/neuro_symbolic.py`

**模块说明**: Hybrid neural-symbolic operators for HelixFactor.

WHY THIS MODULE EXISTS
----------------------
Symbolic expression trees give us interpretability and generalizability, but
they are limited by the vocabulary of hand-coded operators.  Neural leaves
bridge that gap: a tiny MLP trained on historical market data can discover
non-linear interaction patterns (e.g. volume-price divergence under high
intraday volatility) that no single hand-written formula captures.

The workflow is:
  1. Train a NeuralLeaf on historical data to maximise IC with next-period
     returns.  The leaf sees a rolling window of all available features.
  2. Insert the trained leaf into an expression tree as a NeuralLeafNode.
     It behaves like any other operator: (M, T) in -> (M, T) out.
  3. After validation, run distill_to_symbolic() to find the symbolic
     formula from the existing operator library that best approximates
     the neural leaf.  This restores interpretability while keeping the
     discovered signal.
  4. Replace NeuralLeafNode with the distilled formula for production.

Architecture constraints
------------------------
- Each NeuralLeaf has < 5 000 parameters (fits on CPU, fast inference).
- 2-layer MLP: input -> 32 hidden -> 1, with LayerNorm and GELU.
- Input: flattened rolling window of F features over the last W time steps.
- Output: scalar signal per (asset, time) pair, shape (M, T).
- Training uses a differentiable Pearson-IC proxy loss.

#### 类: `DistillationResult`

**说明**: Result of distilling a neural leaf to a symbolic approximation.

##### 方法: `DistillationResult.__str__`


#### 类: `NeuralLeafNode`

**说明**: A node that wraps a NeuralLeaf for use inside expression trees.

##### 方法: `NeuralLeafNode.__init__`


##### 方法: `NeuralLeafNode.evaluate`

- **说明**: Compute the leaf signal on market data.
- **内部/外部调用**: `values`, `evaluate`, `next`, `iter`, `stack`, `full`, `get`

##### 方法: `NeuralLeafNode.to_string`

- **说明**: DSL serialisation.  Returns distilled formula when available.

##### 方法: `NeuralLeafNode.depth`


##### 方法: `NeuralLeafNode.size`


##### 方法: `NeuralLeafNode.clone`

- **内部/外部调用**: `NeuralLeafNode`

##### 方法: `NeuralLeafNode.leaf_features`

- **内部/外部调用**: `sorted`

##### 方法: `NeuralLeafNode.__repr__`

- **内部/外部调用**: `to_string`

##### 方法: `NeuralLeafNode.neural_leaf`


##### 方法: `NeuralLeafNode.set_distilled_formula`

- **说明**: Pin the distilled formula used by ``to_string()``.

#### 类: `SymbolicShell`

**说明**: Wraps a NeuralLeaf as a callable operator compatible with the DSL.

##### 方法: `SymbolicShell.__init__`


##### 方法: `SymbolicShell.__call__`

- **说明**: Evaluate the operator on market data.
- **内部/外部调用**: `evaluate`

##### 方法: `SymbolicShell.formula_string`

- **说明**: Current DSL formula (neural or distilled).
- **内部/外部调用**: `to_string`

##### 方法: `SymbolicShell.is_distilled`


##### 方法: `SymbolicShell.distill`

- **说明**: Run distillation and return the result without modifying state.
- **内部/外部调用**: `distill_to_symbolic`

##### 方法: `SymbolicShell.replace_with_symbolic`

- **说明**: Pin a distilled symbolic formula to this shell.
- **内部/外部调用**: `set_distilled_formula`, `info`

##### 方法: `SymbolicShell.__repr__`


#### 类: `NeuralLeafRegistry`

**说明**: Registry of named, trained NeuralLeaf models.

##### 方法: `NeuralLeafRegistry.__init__`

- **内部/外部调用**: `join`, `gettempdir`, `makedirs`

##### 方法: `NeuralLeafRegistry.register`

- **说明**: Register a trained leaf under *name*.
- **内部/外部调用**: `info`

##### 方法: `NeuralLeafRegistry.get`

- **说明**: Return the leaf registered under *name*, or None.
- **内部/外部调用**: `get`

##### 方法: `NeuralLeafRegistry.remove`

- **说明**: Remove a leaf from the in-memory registry.
- **内部/外部调用**: `pop`

##### 方法: `NeuralLeafRegistry.available`

- **说明**: Return sorted list of registered leaf names.
- **内部/外部调用**: `sorted`, `keys`

##### 方法: `NeuralLeafRegistry._path`

- **内部/外部调用**: `join`, `replace`

##### 方法: `NeuralLeafRegistry.save`

- **说明**: Save a registered leaf's weights to disk.
- **内部/外部调用**: `_path`, `save`, `KeyError`, `state_dict`, `info`, `get`

##### 方法: `NeuralLeafRegistry.load`

- **说明**: Load a leaf from disk and register it.
- **内部/外部调用**: `load_state_dict`, `_path`, `eval`, `FileNotFoundError`, `NeuralLeaf`, `exists`, `info`, `load`

##### 方法: `NeuralLeafRegistry.save_all`

- **说明**: Save all registered leaves.  Returns name -> path mapping.
- **内部/外部调用**: `save`

##### 方法: `NeuralLeafRegistry.load_all`

- **说明**: Load all .pt files from the storage directory.  Returns loaded names.
- **内部/外部调用**: `endswith`, `warning`, `append`, `listdir`, `load`

#### 类: `NeuralLeafConfig`

**说明**: Configuration for a single named neural leaf.

#### 类: `NeuralOperatorIntegration`

**说明**: Orchestrates training, distillation, and persistence of neural leaves.

##### 方法: `NeuralOperatorIntegration.__init__`

- **内部/外部调用**: `NeuralLeafRegistry`

##### 方法: `NeuralOperatorIntegration.train_all_leaves`

- **说明**: Train all listed neural leaves and register them.
- **内部/外部调用**: `warning`, `info`, `register`, `train_neural_leaf`

##### 方法: `NeuralOperatorIntegration.distill_all`

- **说明**: Distill all registered leaves and return name -> best formula.
- **内部/外部调用**: `distill_to_symbolic`, `available`, `info`, `get`

##### 方法: `NeuralOperatorIntegration.get_available_leaves`

- **说明**: Return names of all registered leaves.
- **内部/外部调用**: `available`

##### 方法: `NeuralOperatorIntegration.get_leaf`

- **说明**: Return the NeuralLeaf registered under *name*, or None.
- **内部/外部调用**: `get`

##### 方法: `NeuralOperatorIntegration.get_distillation_result`

- **说明**: Return the stored DistillationResult for *name*, or None.
- **内部/外部调用**: `get`

##### 方法: `NeuralOperatorIntegration.as_node`

- **说明**: Return a NeuralLeafNode ready for use in an expression tree.
- **内部/外部调用**: `get`, `NeuralLeafNode`

##### 方法: `NeuralOperatorIntegration.as_shell`

- **说明**: Return a SymbolicShell for *name*, or None if unknown.
- **内部/外部调用**: `as_node`, `replace_with_symbolic`, `SymbolicShell`

##### 方法: `NeuralOperatorIntegration.save`

- **说明**: Save all registered leaves to *path* (directory).
- **内部/外部调用**: `save_all`, `makedirs`, `available`, `info`

##### 方法: `NeuralOperatorIntegration.load`

- **说明**: Load all .pt files from *path* into the registry.
- **内部/外部调用**: `load_all`, `isdir`, `FileNotFoundError`, `info`

#### 函数: `_build_windows_np`

- **说明**: Create sliding windows from a (M, T, F) array.
- **内部/外部调用**: `as_strided`, `reshape`

#### 函数: `_pearson_ic_loss`

- **说明**: Negative Pearson cross-sectional IC averaged over time steps.
- **内部/外部调用**: `clamp`, `mean`, `std`

#### 函数: `_l2_regularisation`

- **说明**: Compute L2 weight penalty (excludes bias and LayerNorm params).
- **内部/外部调用**: `named_parameters`, `sum`, `tensor`, `pow`

#### 函数: `train_neural_leaf`

- **说明**: Train a NeuralLeaf to maximise cross-sectional IC with next-period returns.
- **内部/外部调用**: `from_numpy`, `eval`, `Adam`, `min`, `any`, `randperm`, `reshape`, `clone`, `no_grad`, `zero_grad`, `CosineAnnealingLR`, `train`, `leaf`, `info`, `param_count`, `parameters`, `backward`, `load_state_dict`, `_pearson_ic_loss`, `max`, `warning`, `state_dict`, `_clean`, `debug`, `step`, `_build_windows_np`, `item`, `device`, `isnan`, `clip_grad_norm_`, `_l2_regularisation`, `items`, `astype`, `range`, `to`, `NeuralLeaf`

#### 函数: `_spearman_corr`

- **说明**: Spearman rank correlation between two flat arrays, ignoring NaN.
- **内部/外部调用**: `sum`, `isnan`, `_spearman`

#### 函数: `_pearson_corr`

- **说明**: Pearson correlation between two flat arrays, ignoring NaN.
- **内部/外部调用**: `mean`, `std`, `sum`, `isnan`

#### 函数: `_evaluate_symbolic_candidate`

- **说明**: Safely evaluate a symbolic candidate, returning None on failure.
- **内部/外部调用**: `values`, `formula_fn`, `next`, `debug`, `iter`

#### 函数: `_build_symbolic_candidates`

- **说明**: Generate all symbolic candidate outputs from the hand-coded operator library.
- **内部/外部调用**: `_ema`, `_safe_add`, `_rolling_apply`, `abs`, `full_like`, `get`

#### 函数: `distill_to_symbolic`

- **说明**: Find the symbolic formula that best approximates the neural leaf.
- **内部/外部调用**: `next`, `iter`, `abs`, `_spearman_corr`, `stack`, `values`, `DistillationResult`, `evaluate`, `full`, `info`, `_build_symbolic_candidates`, `warning`, `max`, `debug`, `sorted`, `_pearson_corr`, `ravel`, `get`, `items`

#### 函数: `get_global_neural_registry`

- **说明**: Return (and lazily create) the global NeuralLeafRegistry.
- **内部/外部调用**: `NeuralLeafRegistry`

#### 函数: `register_neural_leaves_in_operator_registry`

- **说明**: Expose registered neural leaves to the main operator OPERATOR_REGISTRY.
- **内部/外部调用**: `values`, `next`, `evaluate`, `debug`, `get_global_neural_registry`, `OperatorSpec`, `iter`, `stack`, `available`, `_make_np_fn`, `full`, `info`, `get`

#### 函数: `build_default_neural_leaves`

- **说明**: Train the three standard neural leaves on synthetic mock data.
- **内部/外部调用**: `distill_all`, `stack`, `min`, `_pivot`, `NeuralOperatorIntegration`, `generate_mock_data`, `groupby`, `sort_values`, `max`, `NeuralLeafConfig`, `sorted`, `where`, `full_like`, `unique`, `array`, `train_all_leaves`, `MockConfig`, `size`

### 文件: `factorminer/operators/registry.py`

**模块说明**: Central operator registry mapping names to implementations and specs.

Combines the ``OperatorSpec`` definitions from ``core.types`` with the concrete
NumPy / PyTorch function implementations from each category module.

#### 函数: `get_operator`

- **说明**: Look up an operator spec by name.
- **内部/外部调用**: `sorted`, `keys`, `KeyError`

#### 函数: `get_impl`

- **说明**: Return the implementation function for a given operator and backend.
- **内部/外部调用**: `NotImplementedError`, `KeyError`

#### 函数: `execute_operator`

- **说明**: Execute an operator by name.
- **内部/外部调用**: `fn`, `get_impl`

#### 函数: `list_operators`

- **说明**: List all registered operator names.
- **内部/外部调用**: `sorted`, `setdefault`, `keys`, `append`, `sort`, `items`

#### 函数: `implemented_operators`

- **说明**: Return names of operators that have at least a NumPy implementation.
- **内部/外部调用**: `sorted`, `items`

### 文件: `factorminer/operators/regression.py`

**模块说明**: Rolling linear-regression operators.

Each function regresses x against a simple time index [0, 1, ..., window-1]
within a rolling window along axis=1.  Input/output shape: ``(M, T)``.

#### 函数: `_linreg_components_np`

- **说明**: Compute slope, intercept, and fitted values for rolling OLS vs time index.
- **内部/外部调用**: `_pad_front`, `nansum`, `_rolling_np`, `mean`, `nanmean`, `squeeze`, `where`, `full_like`, `sum`, `errstate`, `arange`

#### 函数: `ts_linreg_np`

- **说明**: Rolling linear-regression fitted value.
- **内部/外部调用**: `_linreg_components_np`

#### 函数: `ts_linreg_slope_np`

- **说明**: Rolling linear-regression slope.
- **内部/外部调用**: `_linreg_components_np`

#### 函数: `ts_linreg_intercept_np`

- **说明**: Rolling linear-regression intercept.
- **内部/外部调用**: `_linreg_components_np`

#### 函数: `ts_linreg_resid_np`

- **说明**: Rolling linear-regression residual at the last time step.
- **内部/外部调用**: `_linreg_components_np`

#### 函数: `_linreg_components_torch`

- **说明**: Vectorized rolling OLS on GPU.
- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `unsqueeze`, `mean`, `nanmean`, `_unfold_torch`, `where`, `squeeze`, `sum`, `tensor`, `isnan`, `arange`

#### 函数: `ts_linreg_torch`

- **内部/外部调用**: `_linreg_components_torch`

#### 函数: `ts_linreg_slope_torch`

- **内部/外部调用**: `_linreg_components_torch`

#### 函数: `ts_linreg_intercept_torch`

- **内部/外部调用**: `_linreg_components_torch`

#### 函数: `ts_linreg_resid_torch`

- **内部/外部调用**: `_linreg_components_torch`

### 文件: `factorminer/operators/smoothing.py`

**模块说明**: Moving average / smoothing operators.

Input shape: ``(M, T)`` -> output shape ``(M, T)``.
All operate along the time axis (axis=1) per asset row.

#### 函数: `sma_np`

- **说明**: Simple moving average (identical to Mean).
- **内部/外部调用**: `zeros`, `concatenate`, `nancumsum`, `full_like`

#### 函数: `ema_np`

- **说明**: Exponential moving average with span = window.
- **内部/外部调用**: `copy`, `astype`, `range`, `isnan`

#### 函数: `dema_np`

- **说明**: Double EMA: 2 * EMA(x) - EMA(EMA(x)).
- **内部/外部调用**: `ema_np`

#### 函数: `kama_np`

- **说明**: Kaufman Adaptive Moving Average.
- **内部/外部调用**: `nansum`, `copy`, `where`, `abs`, `astype`, `range`, `errstate`, `isnan`, `diff`

#### 函数: `hma_np`

- **说明**: Hull Moving Average: WMA(2*WMA(x, w/2) - WMA(x, w), sqrt(w)).
- **内部/外部调用**: `max`, `wma_np`, `sqrt`

#### 函数: `sma_torch`

- **说明**: Simple moving average using conv1d for GPU efficiency.
- **内部/外部调用**: `_pad_front_torch`, `_unfold_torch`, `nanmean`

#### 函数: `ema_torch`

- **说明**: EMA -- sequential by nature, but batch across assets.
- **内部/外部调用**: `clone`, `isnan`, `range`

#### 函数: `dema_torch`

- **内部/外部调用**: `ema_torch`

#### 函数: `kama_torch`

- **内部/外部调用**: `nansum`, `where`, `zeros_like`, `abs`, `diff`, `range`, `isnan`, `clone`

#### 函数: `hma_torch`

- **内部/外部调用**: `max`, `wma_torch`

### 文件: `factorminer/operators/statistical.py`

**模块说明**: Rolling-window statistical operators.

Each function operates along the **time** axis (axis=1) independently for
every asset row.  Input shape: ``(M, T)`` -> output shape ``(M, T)``.
The first ``(window - 1)`` values in each row are set to ``NaN``.

#### 函数: `_rolling_np`

- **说明**: Yield views of shape (M, T-w+1, w) using stride tricks.
- **内部/外部调用**: `as_strided`

#### 函数: `_pad_front`

- **说明**: Pad front of time axis with NaN to restore original length.
- **内部/外部调用**: `concatenate`, `full`

#### 函数: `_unfold_torch`

- **说明**: Unfold last dimension to get sliding windows: (M, T) -> (M, T-w+1, w).
- **内部/外部调用**: `unfold`

#### 函数: `_pad_front_torch`

- **内部/外部调用**: `full`, `cat`

#### 函数: `mean_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `nanmean`, `full_like`

#### 函数: `std_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `nanstd`, `full_like`

#### 函数: `var_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `nanvar`, `full_like`

#### 函数: `skew_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `power`, `nanmean`, `squeeze`, `astype`, `full_like`, `sum`, `errstate`, `isnan`

#### 函数: `kurt_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `power`, `nanmean`, `squeeze`, `full_like`, `errstate`

#### 函数: `median_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `full_like`, `nanmedian`

#### 函数: `sum_np`

- **内部/外部调用**: `_pad_front`, `nansum`, `_rolling_np`, `full_like`, `all`, `isnan`

#### 函数: `prod_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `nanprod`, `full_like`, `all`, `isnan`

#### 函数: `ts_max_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `full_like`, `nanmax`

#### 函数: `ts_min_np`

- **内部/外部调用**: `nanmin`, `_pad_front`, `_rolling_np`, `full_like`

#### 函数: `ts_argmax_np`

- **内部/外部调用**: `_pad_front`, `nanargmax`, `_rolling_np`, `astype`, `full_like`

#### 函数: `ts_argmin_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `astype`, `full_like`, `nanargmin`

#### 函数: `ts_rank_np`

- **说明**: Rolling percentile rank of the latest value within its window.
- **内部/外部调用**: `_pad_front`, `nansum`, `_rolling_np`, `astype`, `full_like`, `sum`, `errstate`, `isnan`

#### 函数: `quantile_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `nanquantile`, `full_like`

#### 函数: `count_nan_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `astype`, `full_like`, `sum`, `isnan`

#### 函数: `count_not_nan_np`

- **内部/外部调用**: `_pad_front`, `_rolling_np`, `astype`, `full_like`, `sum`, `isnan`

#### 函数: `mean_torch`

- **内部/外部调用**: `_pad_front_torch`, `_unfold_torch`, `nanmean`

#### 函数: `std_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `nanmean`, `_unfold_torch`, `squeeze`, `clamp`, `sqrt`, `sum`, `isnan`

#### 函数: `var_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `nanmean`, `_unfold_torch`, `squeeze`, `clamp`, `sum`, `isnan`

#### 函数: `skew_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `nanmean`, `_unfold_torch`, `squeeze`, `clamp`, `pow`, `sum`, `isnan`

#### 函数: `kurt_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `nanmean`, `_unfold_torch`, `squeeze`, `clamp`, `pow`, `sum`, `isnan`

#### 函数: `median_torch`

- **内部/外部调用**: `_pad_front_torch`, `_unfold_torch`, `nanmedian`

#### 函数: `sum_torch`

- **内部/外部调用**: `nansum`, `_pad_front_torch`, `_unfold_torch`, `all`, `isnan`

#### 函数: `prod_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `prod`, `_unfold_torch`, `all`, `isnan`

#### 函数: `ts_max_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `max`, `_unfold_torch`, `all`, `isnan`

#### 函数: `ts_min_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `_unfold_torch`, `min`, `all`, `isnan`

#### 函数: `ts_argmax_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `_unfold_torch`, `argmax`

#### 函数: `ts_argmin_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `_unfold_torch`, `argmin`

#### 函数: `ts_rank_torch`

- **说明**: Rolling percentile rank -- key GPU acceleration target (17x speedup).
- **内部/外部调用**: `_pad_front_torch`, `_unfold_torch`, `clamp`, `sum`, `isnan`

#### 函数: `quantile_torch`

- **内部/外部调用**: `long`, `_pad_front_torch`, `unsqueeze`, `squeeze`, `_unfold_torch`, `clamp`, `gather`, `sum`, `sort`, `isnan`, `nanmedian`

#### 函数: `count_nan_torch`

- **内部/外部调用**: `_pad_front_torch`, `_unfold_torch`, `sum`, `isnan`

#### 函数: `count_not_nan_torch`

- **内部/外部调用**: `_pad_front_torch`, `_unfold_torch`, `sum`, `isnan`

### 文件: `factorminer/operators/timeseries.py`

**模块说明**: Time-series operators along the T axis for each asset row.

Input shape: ``(M, T)`` -> output shape ``(M, T)``.

#### 函数: `delta_np`

- **说明**: x[t] - x[t - period].
- **内部/外部调用**: `full_like`

#### 函数: `delay_np`

- **说明**: x[t - period] (lag operator).
- **内部/外部调用**: `full_like`

#### 函数: `return_np`

- **说明**: x[t] / x[t-d] - 1.
- **内部/外部调用**: `full_like`, `abs`

#### 函数: `log_return_np`

- **说明**: log(x[t] / x[t-d]).
- **内部/外部调用**: `where`, `abs`, `log`, `full_like`, `errstate`

#### 函数: `corr_np`

- **说明**: Rolling Pearson correlation.
- **内部/外部调用**: `_pad_front`, `_rolling_np`, `nanmean`, `where`, `sqrt`, `full_like`, `errstate`

#### 函数: `cov_np`

- **说明**: Rolling covariance.
- **内部/外部调用**: `_pad_front`, `_rolling_np`, `nanmean`, `full_like`

#### 函数: `beta_np`

- **说明**: Rolling regression beta: slope of x regressed on y.
- **内部/外部调用**: `_pad_front`, `_rolling_np`, `nanmean`, `where`, `full_like`, `errstate`

#### 函数: `resid_np`

- **说明**: Rolling regression residual: x - beta * y - alpha, evaluated at last point.
- **内部/外部调用**: `_pad_front`, `_rolling_np`, `nanmean`, `squeeze`, `where`, `full_like`, `errstate`

#### 函数: `wma_np`

- **说明**: Linearly weighted moving average.
- **内部/外部调用**: `nansum`, `_pad_front`, `_rolling_np`, `full_like`, `sum`, `arange`

#### 函数: `decay_np`

- **说明**: Exponentially decaying sum (linearly decaying weighted average).
- **内部/外部调用**: `wma_np`

#### 函数: `cumsum_np`

- **内部/外部调用**: `nancumsum`

#### 函数: `cumprod_np`

- **内部/外部调用**: `cumprod`, `where`, `isnan`

#### 函数: `cummax_np`

- **内部/外部调用**: `fmax`, `copy`, `range`

#### 函数: `cummin_np`

- **内部/外部调用**: `fmin`, `copy`, `range`

#### 函数: `delta_torch`

- **内部/外部调用**: `full_like`

#### 函数: `delay_torch`

- **内部/外部调用**: `full_like`

#### 函数: `return_torch`

- **内部/外部调用**: `full_like`, `abs`

#### 函数: `log_return_torch`

- **内部/外部调用**: `abs`, `log`, `full_like`

#### 函数: `corr_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `nanmean`, `_unfold_torch`, `where`, `sqrt`, `clamp`, `sum`, `tensor`, `isnan`

#### 函数: `cov_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `nanmean`, `_unfold_torch`, `clamp`, `sum`, `isnan`

#### 函数: `beta_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `nanmean`, `_unfold_torch`, `where`, `clamp`, `sum`, `tensor`, `isnan`

#### 函数: `resid_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `nanmean`, `_unfold_torch`, `where`, `squeeze`, `clamp`, `zeros_like`, `sum`, `isnan`

#### 函数: `wma_torch`

- **内部/外部调用**: `nan_to_num`, `_pad_front_torch`, `unsqueeze`, `_unfold_torch`, `sum`, `arange`

#### 函数: `decay_torch`

- **内部/外部调用**: `wma_torch`

#### 函数: `cumsum_torch`

- **内部/外部调用**: `nan_to_num`, `cumsum`

#### 函数: `cumprod_torch`

- **内部/外部调用**: `nan_to_num`, `cumprod`

#### 函数: `cummax_torch`

- **内部/外部调用**: `nan_to_num`, `cummax`

#### 函数: `cummin_torch`

- **内部/外部调用**: `nan_to_num`, `cummin`

### 文件: `factorminer/utils/__init__.py`

**模块说明**: Utility modules for FactorMiner.

### 文件: `factorminer/utils/config.py`

**模块说明**: Configuration loading, validation, and management for FactorMiner.

#### 类: `MiningConfig`

**说明**: Parameters controlling the factor mining loop.

##### 方法: `MiningConfig.validate`

- **内部/外部调用**:

#### 类: `EvaluationConfig`

**说明**: Parameters for factor evaluation.

##### 方法: `EvaluationConfig.validate`

- **内部/外部调用**:

#### 类: `DataConfig`

**说明**: Parameters for data loading and universes.

##### 方法: `DataConfig.validate`

- **内部/外部调用**: `strip`, `append`, `get`

#### 类: `LLMConfig`

**说明**: Parameters for LLM-based factor generation.

##### 方法: `LLMConfig.validate`

- **内部/外部调用**:

#### 类: `MemoryConfig`

**说明**: Parameters for the experience memory system.

##### 方法: `MemoryConfig.validate`

- **内部/外部调用**:

#### 类: `CausalConfig`

**说明**: Parameters for causal validation (Granger + intervention tests).

##### 方法: `CausalConfig.validate`

- **内部/外部调用**: `abs`

#### 类: `RegimeConfig`

**说明**: Parameters for regime-conditional factor evaluation.

##### 方法: `RegimeConfig.validate`

- **内部/外部调用**:

#### 类: `CapacityConfig`

**说明**: Parameters for strategy capacity estimation.

##### 方法: `CapacityConfig.validate`

- **内部/外部调用**:

#### 类: `SignificanceConfig`

**说明**: Parameters for statistical significance testing.

##### 方法: `SignificanceConfig.validate`

- **内部/外部调用**:

#### 类: `DebateConfig`

**说明**: Parameters for multi-specialist debate-based generation.

##### 方法: `DebateConfig.validate`

- **内部/外部调用**:

#### 类: `AutoInventorConfig`

**说明**: Parameters for automatic operator invention.

##### 方法: `AutoInventorConfig.validate`

- **内部/外部调用**:

#### 类: `HelixConfig`

**说明**: Parameters for the Helix knowledge and memory system.

##### 方法: `HelixConfig.validate`

- **内部/外部调用**:

#### 类: `Phase2Config`

**说明**: Aggregated configuration for all Phase 2 subsystems.

##### 方法: `Phase2Config.validate`

- **内部/外部调用**: `validate`

#### 类: `BenchmarkConfig`

**说明**: Parameters for paper/research benchmark execution.

##### 方法: `BenchmarkConfig.validate`

- **内部/外部调用**: `any`

#### 类: `ResearchUncertaintyConfig`

**说明**: Uncertainty controls for multi-horizon research scoring.

##### 方法: `ResearchUncertaintyConfig.validate`

- **内部/外部调用**:

#### 类: `ResearchAdmissionConfig`

**说明**: Research-mode admission controls.

##### 方法: `ResearchAdmissionConfig.validate`

- **内部/外部调用**:

#### 类: `ResearchSelectionConfig`

**说明**: Research-mode model configuration.

##### 方法: `ResearchSelectionConfig.validate`

- **内部/外部调用**: `any`

#### 类: `ResearchRegimesConfig`

**说明**: Research-mode regime diagnostics.

##### 方法: `ResearchRegimesConfig.validate`

- **内部/外部调用**:

#### 类: `ResearchExecutionConfig`

**说明**: Execution-aware research scoring controls.

##### 方法: `ResearchExecutionConfig.validate`

- **内部/外部调用**:

#### 类: `ResearchConfig`

**说明**: Research-first multi-horizon scoring configuration.

##### 方法: `ResearchConfig.validate`

- **内部/外部调用**: `values`, `validate`, `any`

#### 类: `Config`

**说明**: Top-level configuration aggregating all sub-configs.

##### 方法: `Config.validate`

- **说明**: Validate all sub-configurations.
- **内部/外部调用**: `validate`

##### 方法: `Config.to_dict`

- **说明**: Serialize config to a plain dictionary.
- **内部/外部调用**: `asdict`

##### 方法: `Config.save`

- **说明**: Write config to a YAML file.
- **内部/外部调用**: `dump`, `open`, `to_dict`, `mkdir`, `Path`

#### 函数: `_deep_merge`

- **说明**: Recursively merge override into base, returning a new dict.
- **内部/外部调用**: `deepcopy`, `_deep_merge`, `items`

#### 函数: `_load_yaml`

- **说明**: Load a YAML file and return its contents as a dict.
- **内部/外部调用**: `open`, `safe_load`

#### 函数: `_build_section`

- **说明**: Instantiate a config dataclass, ignoring unknown keys.
- **内部/外部调用**: `values`, `section_cls`, `items`

#### 函数: `_build_phase2`

- **说明**: Build Phase2Config with nested sub-config dataclasses.
- **内部/外部调用**: `get`, `_build_section`, `Phase2Config`, `items`, `sub_cls`

#### 函数: `_build_research`

- **说明**: Build ResearchConfig with nested sub-config dataclasses.
- **内部/外部调用**: `get`, `values`, `deepcopy`, `_build_section`, `ResearchConfig`, `items`, `sub_cls`

#### 函数: `load_config`

- **说明**: Load configuration from YAML with defaults and optional overrides.
- **内部/外部调用**: `get`, `_load_yaml`, `Config`, `validate`, `_deep_merge`, `_build_section`, `_build_research`, `_build_phase2`, `items`, `Path`

### 文件: `factorminer/utils/logging.py`

**模块说明**: Structured logging system for FactorMiner mining sessions.

#### 类: `FactorRecord`

**说明**: Log record for a single evaluated factor candidate.

##### 方法: `FactorRecord.to_dict`

- **内部/外部调用**: `items`, `asdict`

#### 类: `IterationRecord`

**说明**: Aggregated stats for a single mining iteration (batch).

##### 方法: `IterationRecord.yield_rate`

- **说明**: Fraction of candidates that were admitted to the library.

##### 方法: `IterationRecord.to_dict`

- **内部/外部调用**: `asdict`

#### 类: `JSONLogExporter`

**说明**: Collects structured records and exports them to a JSON file.

##### 方法: `JSONLogExporter.__init__`


##### 方法: `JSONLogExporter.add_iteration`

- **内部/外部调用**: `to_dict`, `append`

##### 方法: `JSONLogExporter.add_factor`

- **内部/外部调用**: `to_dict`, `append`

##### 方法: `JSONLogExporter.export`

- **内部/外部调用**: `_summary`, `dump`, `open`, `mkdir`, `Path`

##### 方法: `JSONLogExporter._summary`

- **内部/外部调用**: `sum`, `get`

#### 类: `_ConsoleFormatter`

**说明**: Compact colored formatter for terminal output.

##### 方法: `_ConsoleFormatter.format`

- **内部/外部调用**: `strftime`, `localtime`, `get`, `getMessage`

#### 类: `MiningSessionLogger`

**说明**: High-level logger for an entire mining session.

##### 方法: `MiningSessionLogger.__init__`

- **内部/外部调用**: `setup_logger`, `JSONLogExporter`, `mkdir`, `Path`

##### 方法: `MiningSessionLogger.start_progress`

- **内部/外部调用**: `tqdm`

##### 方法: `MiningSessionLogger.advance_progress`

- **内部/外部调用**: `update`

##### 方法: `MiningSessionLogger.close_progress`

- **内部/外部调用**: `close`

##### 方法: `MiningSessionLogger.log_iteration`

- **说明**: Log a completed iteration to both console and structured store.
- **内部/外部调用**: `info`, `add_iteration`, `advance_progress`

##### 方法: `MiningSessionLogger.log_factor`

- **说明**: Log a single factor evaluation result.
- **内部/外部调用**: `debug`, `add_factor`

##### 方法: `MiningSessionLogger.log_session_start`

- **内部/外部调用**: `info`, `get`

##### 方法: `MiningSessionLogger.log_session_end`

- **内部/外部调用**: `_summary`, `export`, `close_progress`, `info`, `get`

#### 函数: `setup_logger`

- **说明**: Create and configure a FactorMiner logger.
- **内部/外部调用**: `Formatter`, `StreamHandler`, `setFormatter`, `_ConsoleFormatter`, `addHandler`, `mkdir`, `getLogger`, `setLevel`, `clear`, `FileHandler`, `Path`

### 文件: `factorminer/utils/reporting.py`

**模块说明**: Mining session reporting for FactorMiner.

Provides structured logging, text reports, JSON export, and progress
visualization for factor mining sessions. Designed to mirror the batch
reports shown in Appendix H of the paper.

#### 类: `FactorAdmissionRecord`

**说明**: Record of a single factor admission.

##### 方法: `FactorAdmissionRecord.__post_init__`

- **内部/外部调用**: `now`, `strftime`

#### 类: `BatchRecord`

**说明**: Record of a single mining batch.

##### 方法: `BatchRecord.__post_init__`

- **内部/外部调用**: `now`, `strftime`

##### 方法: `BatchRecord.rejected`


##### 方法: `BatchRecord.yield_rate`


##### 方法: `BatchRecord.rejection_rate`


##### 方法: `BatchRecord.to_dict`

- **内部/外部调用**: `asdict`

#### 类: `MiningReporter`

**说明**: Track and report mining session progress.

##### 方法: `MiningReporter.__init__`

- **内部/外部调用**: `time`, `mkdir`, `Path`

##### 方法: `MiningReporter.log_batch`

- **说明**: Log a batch's results.
- **内部/外部调用**: `append`, `BatchRecord`

##### 方法: `MiningReporter.log_factor_admission`

- **说明**: Log an individual factor admission.
- **内部/外部调用**: `FactorAdmissionRecord`, `append`

##### 方法: `MiningReporter.generate_batch_report`

- **说明**: Generate text report for a specific batch.
- **内部/外部调用**: `defaultdict`, `join`, `lower`, `sorted`, `upper`, `append`, `items`

##### 方法: `MiningReporter.generate_session_report`

- **说明**: Generate full session report with cumulative statistics.
- **内部/外部调用**: `now`, `join`, `strftime`, `sorted`, `append`, `time`, `sum`

##### 方法: `MiningReporter.export_to_json`

- **说明**: Export all mining logs to JSON.
- **内部/外部调用**: `dump`, `asdict`, `fromtimestamp`, `_compute_summary`, `open`, `strftime`, `to_dict`, `mkdir`, `time`, `Path`

##### 方法: `MiningReporter.save_session_report`

- **说明**: Save the session report to a text file.
- **内部/外部调用**: `generate_session_report`, `open`, `write`

##### 方法: `MiningReporter.plot_mining_progress`

- **说明**: Plot library growth, yield rate, and rejection rate over batches.
- **内部/外部调用**: `set_title`, `fill_between`, `bar`, `set_xlabel`, `zip`, `subplots`, `set_ylim`, `show`, `close`, `set_ylabel`, `update`, `tight_layout`, `legend`, `sum`, `plot`, `savefig`, `text`, `axhline`

##### 方法: `MiningReporter._compute_summary`

- **说明**: Compute cumulative summary statistics.
- **内部/外部调用**: `sum`, `time`

### 文件: `factorminer/utils/tearsheet.py`

**模块说明**: Factor tear sheet generation for FactorMiner.

Produces comprehensive, multi-panel evaluation reports for individual
factors, following the style of Appendix O / Figure 10 from the paper.
Also provides summary table generation for the full factor library.

#### 类: `FactorTearSheet`

**说明**: Generate comprehensive evaluation report for a single factor.

##### 方法: `FactorTearSheet.generate`

- **说明**: Generate a multi-panel tear sheet.
- **内部/外部调用**: `compute_ic`, `choice`, `set_title`, `fill_between`, `cmap`, `std`, `any`, `compute_icir`, `_rolling_mean`, `bar`, `RdYlGn`, `percentile`, `get_height`, `set_xlabel`, `zip`, `_compute_daily_turnover`, `compute_turnover`, `compute_ic_mean`, `figure`, `default_rng`, `compute_ic_win_rate`, `show`, `append`, `close`, `_set_date_ticks`, `set_ylabel`, `text`, `cumsum`, `get_x`, `max`, `where`, `update`, `tight_layout`, `nancumsum`, `clip`, `linspace`, `compute_quintile_returns`, `legend`, `suptitle`, `sum`, `get`, `enumerate`, `arange`, `isnan`, `rankdata`, `axvline`, `ceil`, `mean`, `hist`, `astype`, `plot`, `savefig`, `add_subplot`, `range`, `get_width`, `axhline`, `GridSpec`

##### 方法: `FactorTearSheet.generate_summary_table`

- **说明**: Generate summary table for all factors in the library.
- **内部/外部调用**: `reset_index`, `append`, `DataFrame`, `get`, `sort_values`

##### 方法: `FactorTearSheet._set_date_ticks`

- **说明**: Set evenly spaced date tick labels on the x-axis.
- **内部/外部调用**: `max`, `set_xticks`, `min`, `range`, `set_xticklabels`

#### 函数: `_rolling_mean`

- **说明**: Compute rolling mean with edge handling.
- **内部/外部调用**: `mean`, `where`, `convolve`, `ones`, `full_like`, `range`, `isnan`

#### 函数: `_compute_daily_turnover`

- **说明**: Compute total daily turnover as fraction of positions changing.
- **内部/外部调用**: `rankdata`, `mean`, `abs`, `full`, `range`, `sum`, `isnan`

### 文件: `factorminer/utils/visualization.py`

**模块说明**: Core visualization functions for FactorMiner.

Provides publication-quality plots for factor analysis, mining diagnostics,
and performance reporting. Uses matplotlib and seaborn with a consistent
style inspired by the FactorMiner paper figures.

#### 函数: `_apply_style`

- **说明**: Apply a clean, publication-quality matplotlib style once.
- **内部/外部调用**: `update`

#### 函数: `_save_or_show`

- **说明**: Save figure to disk or display interactively.
- **内部/外部调用**: `show`, `savefig`, `close`

#### 函数: `plot_correlation_heatmap`

- **说明**: Generate pairwise Spearman correlation heatmap.
- **内部/外部调用**: `subplots`, `max`, `set_title`, `tick_params`, `_apply_style`, `zeros_like`, `heatmap`, `nanmean`, `tight_layout`, `abs`, `fill_diagonal`, `min`, `_save_or_show`, `triu_indices`

#### 函数: `plot_ic_timeseries`

- **说明**: Plot IC time series with rolling average and cumulative IC.
- **内部/外部调用**: `set_title`, `fill_between`, `_apply_style`, `convolve`, `set_xticks`, `min`, `bar`, `set_xlabel`, `subplots`, `set_ylabel`, `_save_or_show`, `max`, `where`, `tight_layout`, `nancumsum`, `ones`, `legend`, `isnan`, `arange`, `mean`, `nanmean`, `plot`, `range`, `axhline`, `set_xticklabels`

#### 函数: `plot_quintile_returns`

- **说明**: Plot Q1-Q5 quintile bar chart and cumulative returns.
- **内部/外部调用**: `set_title`, `cmap`, `join`, `_apply_style`, `bar`, `get_height`, `set_xlabel`, `zip`, `startswith`, `subplots`, `keys`, `append`, `set_ylabel`, `_save_or_show`, `text`, `get_x`, `max`, `sorted`, `tight_layout`, `legend`, `plot`, `range`, `get_width`, `axhline`

#### 函数: `plot_ablation_comparison`

- **说明**: Bar charts comparing Have Memory vs No Memory ablation.
- **内部/外部调用**: `subplots`, `get`, `get_x`, `set_title`, `_apply_style`, `get_width`, `set_xticks`, `set_ylabel`, `tight_layout`, `legend`, `text`, `bar`, `_save_or_show`, `get_height`, `set_xticklabels`, `arange`

#### 函数: `plot_efficiency_benchmark`

- **说明**: Grouped bar chart on log scale for computation time.
- **内部/外部调用**: `set_title`, `_apply_style`, `set_xticks`, `bar`, `get_height`, `zip`, `subplots`, `values`, `set_major_formatter`, `set_yscale`, `keys`, `set_ylabel`, `_save_or_show`, `get_x`, `max`, `sorted`, `tight_layout`, `legend`, `ScalarFormatter`, `enumerate`, `arange`, `get`, `text`, `get_width`, `set_xticklabels`

#### 函数: `plot_cost_pressure`

- **说明**: Cumulative return plots under different transaction cost settings.
- **内部/外部调用**: `viridis`, `set_title`, `_apply_style`, `min`, `set_xlabel`, `zip`, `subplots`, `set_yscale`, `keys`, `set_ylabel`, `_save_or_show`, `sorted`, `tight_layout`, `linspace`, `legend`, `plot`, `axhline`, `asarray`

#### 函数: `plot_mining_funnel`

- **说明**: Funnel chart showing Stage 1 -> 2 -> 3 -> 4 filtering.
- **内部/外部调用**: `set_title`, `_apply_style`, `set_xticks`, `zip`, `subplots`, `set_ylim`, `get_yaxis_transform`, `set_visible`, `_save_or_show`, `text`, `set_xlim`, `max`, `set_yticks`, `barh`, `tight_layout`, `get`, `enumerate`, `range`
