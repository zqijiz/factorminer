import ast
import os
import glob
from collections import defaultdict

def extract_info():
    project_root = "factorminer"
    structure = {}

    for root, dirs, files in os.walk(project_root):
        if "tests" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = ast.parse(source, filename=file_path)
                except Exception as e:
                    print(f"Failed to parse {file_path}: {e}")
                    continue

                module_info = {
                    "docstring": ast.get_docstring(tree),
                    "classes": {},
                    "functions": {}
                }

                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.ClassDef):
                        class_info = {
                            "docstring": ast.get_docstring(node),
                            "methods": {}
                        }
                        for class_node in node.body:
                            if isinstance(class_node, ast.FunctionDef):
                                calls = [c.func.id for c in ast.walk(class_node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
                                attrs = [c.func.attr for c in ast.walk(class_node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)]
                                class_info["methods"][class_node.name] = {
                                    "docstring": ast.get_docstring(class_node),
                                    "calls": list(set(calls + attrs))
                                }
                        module_info["classes"][node.name] = class_info
                    elif isinstance(node, ast.FunctionDef):
                        calls = [c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
                        attrs = [c.func.attr for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)]
                        module_info["functions"][node.name] = {
                            "docstring": ast.get_docstring(node),
                            "calls": list(set(calls + attrs))
                        }

                structure[file_path] = module_info

    return structure

def generate_markdown(structure):
    md = """# FactorMiner 项目架构与算法详细文档

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

"""

    for file_path, info in sorted(structure.items()):
        md += f"### 文件: `{file_path}`\n\n"
        if info["docstring"]:
            # truncate docstring if too long
            doc = info['docstring'].strip()
            md += f"**模块说明**: {doc}\n\n"

        if info["classes"]:
            for cls_name, cls_info in info["classes"].items():
                md += f"#### 类: `{cls_name}`\n\n"
                if cls_info["docstring"]:
                    doc = cls_info['docstring'].split('\n')[0].strip() # keep it short
                    md += f"**说明**: {doc}\n\n"

                for method_name, method_info in cls_info["methods"].items():
                    md += f"##### 方法: `{cls_name}.{method_name}`\n\n"
                    if method_info["docstring"]:
                        doc = method_info['docstring'].split('\n')[0].strip() # keep it short
                        md += f"- **说明**: {doc}\n"
                    if method_info["calls"]:
                        md += f"- **内部/外部调用**: {', '.join('`' + c + '`' for c in method_info['calls'] if c not in ['str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple', 'len', 'print', 'isinstance', 'getattr', 'setattr', 'hasattr', 'super', 'Exception', 'ValueError', 'TypeError', 'RuntimeError'])}\n\n"
                    else:
                        md += "\n"

        if info["functions"]:
            for func_name, func_info in info["functions"].items():
                md += f"#### 函数: `{func_name}`\n\n"
                if func_info["docstring"]:
                    doc = func_info['docstring'].split('\n')[0].strip() # keep it short
                    md += f"- **说明**: {doc}\n"
                if func_info["calls"]:
                    md += f"- **内部/外部调用**: {', '.join('`' + c + '`' for c in func_info['calls'] if c not in ['str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple', 'len', 'print', 'isinstance', 'getattr', 'setattr', 'hasattr', 'super', 'Exception', 'ValueError', 'TypeError', 'RuntimeError'])}\n\n"
                else:
                    md += "\n"

    return md

structure = extract_info()
md_content = generate_markdown(structure)
with open("Detailed_Architecture_zh.md", "w", encoding="utf-8") as f:
    f.write(md_content)

print("Documentation generated at Detailed_Architecture_zh.md")
