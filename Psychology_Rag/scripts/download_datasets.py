#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集一键下载脚本。

运行前请先安装依赖：
  pip install datasets huggingface_hub requests tqdm

用法：
  python scripts/download_datasets.py

脚本会按优先级依次下载所有可自动获取的数据集，
并在 datasets/ 目录下创建对应的子目录。
"""

import os
import sys
import json
from pathlib import Path

# ── 项目根目录 & 数据存放目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def ensure_dir(path: Path):
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# 1. PsyQA (清华 thu-coai) — 从 HuggingFace 镜像下载
# ============================================================================
def download_psyqa():
    """
    下载 PsyQA 数据集。

    来源: HuggingFace lsy641/PsyQA（已有 train/valid/test 划分）
    原始: thu-coai/PsyQA (ACL 2021)
    """
    print("\n" + "=" * 60)
    print("  📥 [1/5] 下载 PsyQA 数据集...")
    print("  来源: HuggingFace lsy641/PsyQA")
    print("=" * 60)

    target_dir = ensure_dir(DATASETS_DIR / "psyqa")

    # 检查是否已下载
    if (target_dir / "train.json").exists():
        print("  ⏭️  已存在，跳过下载。如需重新下载请先删除 datasets/psyqa/")
        return True

    try:
        from datasets import load_dataset

        print("  正在从 HuggingFace 加载 lsy641/PsyQA ...")
        ds = load_dataset("lsy641/PsyQA")

        # 保存各分片
        for split_name, file_name in [
            ("train", "train.json"),
            ("validation", "valid.json"),
            ("test", "test.json"),
        ]:
            if split_name in ds:
                output_path = target_dir / file_name
                ds[split_name].to_json(str(output_path), force_ascii=False)
                print(f"  ✅ {split_name}: {len(ds[split_name])} 条 → {file_name}")

        print("  🎉 PsyQA 下载完成!")
        return True

    except ImportError:
        print("  ❌ 需要安装 datasets 库: pip install datasets")
        return False
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        print("  💡 备选方案: 访问 https://huggingface.co/datasets/lsy641/PsyQA 手动下载")
        print("     或向原作者申请: https://github.com/thu-coai/PsyQA")
        return False


# ============================================================================
# 2. CPsyCounD (CAS-SIAT-XinHai) — 从 HuggingFace 下载
# ============================================================================
def download_cpsycoun():
    """
    下载 CPsyCounD 数据集。

    来源: HuggingFace CAS-SIAT-XinHai/CPsyCoun
    论文: ACL 2024 Findings
    """
    print("\n" + "=" * 60)
    print("  📥 [2/5] 下载 CPsyCounD 数据集...")
    print("  来源: HuggingFace CAS-SIAT-XinHai/CPsyCoun")
    print("=" * 60)

    target_dir = ensure_dir(DATASETS_DIR / "cpsycoun")

    if (target_dir / "CPsyCounD.json").exists():
        print("  ⏭️  已存在，跳过下载。")
        return True

    try:
        from datasets import load_dataset

        print("  正在从 HuggingFace 加载 CAS-SIAT-XinHai/CPsyCoun ...")
        ds = load_dataset("CAS-SIAT-XinHai/CPsyCoun")

        # 保存
        split = "train" if "train" in ds else list(ds.keys())[0]
        output_path = target_dir / "CPsyCounD.json"
        ds[split].to_json(str(output_path), force_ascii=False)
        print(f"  ✅ {len(ds[split])} 条多轮对话 → CPsyCounD.json")

        print("  🎉 CPsyCounD 下载完成!")
        return True

    except ImportError:
        print("  ❌ 需要安装 datasets 库: pip install datasets")
        return False
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        print("  💡 备选方案:")
        print("     1. 访问 https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyCoun")
        print("     2. 或从 GitHub 克隆: git clone https://github.com/CAS-SIAT-XinHai/CPsyCoun")
        return False


# ============================================================================
# 3. WHO mhGAP-IG V2.0 — 从 WHO 官网下载 PDF
# ============================================================================
def download_mhgap():
    """
    下载 WHO mhGAP 干预指南 V2.0。

    来源: WHO IRIS (官方开放获取)
    许可: CC BY-NC-SA 3.0 IGO
    """
    print("\n" + "=" * 60)
    print("  📥 [3/5] 下载 WHO mhGAP-IG V2.0...")
    print("  来源: WHO 官方出版物")
    print("=" * 60)

    target_dir = ensure_dir(DATASETS_DIR / "mhgap")
    output_path = target_dir / "mhgap-ig-v2.pdf"

    if output_path.exists():
        print("  ⏭️  已存在，跳过下载。")
        return True

    url = "https://iris.who.int/bitstream/handle/10665/250239/9789241549790-eng.pdf?sequence=1"

    try:
        import requests
        from tqdm import tqdm

        print(f"  正在下载 PDF (约 5MB)...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="  mhGAP-IG") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"  ✅ 已保存到 {output_path}")
        print("  🎉 mhGAP-IG 下载完成!")
        return True

    except ImportError:
        print("  ❌ 需要安装: pip install requests tqdm")
        return False
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        print("  💡 手动下载:")
        print("     访问 https://www.who.int/publications/i/item/9789241549790")
        print(f"     保存到 {output_path}")
        return False


# ============================================================================
# 4. OpenStax Psychology 2e — 从 OpenStax 下载 PDF
# ============================================================================
def download_openstax():
    """
    下载 OpenStax Psychology 2e 教材。

    来源: OpenStax (Rice University)
    许可: CC BY 4.0，完全免费
    """
    print("\n" + "=" * 60)
    print("  📥 [4/5] 下载 OpenStax Psychology 2e...")
    print("  来源: OpenStax (CC BY 4.0)")
    print("=" * 60)

    target_dir = ensure_dir(DATASETS_DIR / "openstax_psychology")
    output_path = target_dir / "Psychology2e-WEB.pdf"

    if output_path.exists():
        print("  ⏭️  已存在，跳过下载。")
        return True

    url = "https://assets.openstax.org/oscms-prodcms/media/documents/Psychology2e-WEB.pdf"

    try:
        import requests
        from tqdm import tqdm

        print(f"  正在下载 PDF (约 50MB，请耐心等待)...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="  Psychology2e") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"  ✅ 已保存到 {output_path}")
        print("  🎉 OpenStax Psychology 2e 下载完成!")
        return True

    except ImportError:
        print("  ❌ 需要安装: pip install requests tqdm")
        return False
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        print("  💡 手动下载:")
        print("     访问 https://openstax.org/details/books/psychology-2e")
        print(f"     保存到 {output_path}")
        return False


# ============================================================================
# 5. 安全护栏科普文档 — 自动生成模板
# ============================================================================
def create_guardrail_templates():
    """
    创建 PHQ-9 / GAD-7 科普说明模板文档。

    ⚠️ 这些不是量表原文，仅为科普说明！
    """
    print("\n" + "=" * 60)
    print("  📝 [5/5] 创建安全护栏科普文档模板...")
    print("=" * 60)

    target_dir = ensure_dir(DATASETS_DIR / "guardrails")

    # ── PHQ-9 科普说明 ──
    phq9_path = target_dir / "phq9_intro.txt"
    if not phq9_path.exists():
        phq9_content = """PHQ-9 科普说明

PHQ-9（患者健康问卷-9）是一种被广泛使用的抑郁症状自我筛查工具。它由 9 个问题组成，涵盖情绪、兴趣、睡眠、精力、食欲、自我评价、注意力、行动力和自伤想法等维度。PHQ-9 最初由 Drs. Robert L. Spitzer, Janet B.W. Williams, Kurt Kroenke 及同事于 1999 年开发，目的是为初级医疗保健场景提供简便的抑郁症状筛查。

PHQ-9 的主要用途：
在初级医疗保健场景中初步筛查抑郁症状；帮助医疗专业人员了解患者近两周的情绪状态；作为研究工具评估群体抑郁症状的严重程度。

需要特别强调的局限性：
PHQ-9 的结果不能替代专业诊断。任何筛查结果都应由具有资质的精神科医生或临床心理师进行综合评估后才能作为参考。不建议个人在没有专业指导的情况下仅凭分数做出判断。量表有文化适应性的局限，不同文化背景下可能存在差异。

如果你在 PHQ-9 筛查中发现自己可能存在抑郁倾向，最重要的下一步是寻求专业心理咨询师或精神科医生的帮助，而不是仅凭分数给自己"下诊断"。"""
        phq9_path.write_text(phq9_content, encoding="utf-8")
        print(f"  ✅ 已创建 PHQ-9 科普说明 → {phq9_path}")
    else:
        print("  ⏭️  PHQ-9 文档已存在")

    # ── GAD-7 科普说明 ──
    gad7_path = target_dir / "gad7_intro.txt"
    if not gad7_path.exists():
        gad7_content = """GAD-7 科普说明

GAD-7（广泛性焦虑障碍量表-7）是一种焦虑症状的简短自我筛查工具，包含 7 个问题，评估个体在过去两周内焦虑症状的频率和严重程度。该量表由 Spitzer, Kroenke, Williams 和 Löwe 于 2006 年开发。

GAD-7 的主要用途：
在社区健康筛查中初步识别可能需要进一步评估的个体；在临床研究中评估焦虑症状的严重程度；辅助医疗专业人员进行初步的焦虑症状评估。

需要特别强调的局限性：
GAD-7 仅为筛查工具，不能作为确诊依据。具体诊断需要由专业人员通过临床访谈和综合评估来完成。焦虑症状可能与多种心理和躯体疾病重叠，需要专业鉴别。该量表主要针对广泛性焦虑，对其他类型焦虑障碍（如社交焦虑障碍、惊恐障碍）的敏感性有限。

如果你在 GAD-7 筛查中发现自己可能存在焦虑倾向，建议及时寻求专业心理咨询师或精神科医生的评估和帮助。"""
        gad7_path.write_text(gad7_content, encoding="utf-8")
        print(f"  ✅ 已创建 GAD-7 科普说明 → {gad7_path}")
    else:
        print("  ⏭️  GAD-7 文档已存在")

    # ── 危机干预指引 ──
    crisis_path = target_dir / "crisis_intervention.txt"
    if not crisis_path.exists():
        crisis_content = """心理危机识别与转介指引

如果你或你认识的人正在经历以下情况，请立即寻求专业帮助：

需要紧急关注的信号：
（1）出现自伤或自杀的想法或计划
（2）反复出现伤害他人的念头
（3）持续两周以上无法正常工作、学习或社交
（4）出现幻觉、妄想等严重的感知异常
（5）严重的物质滥用问题（酒精、药物等）

中国大陆心理援助热线：
全国 24 小时心理援助热线: 400-161-9995
北京心理危机研究与干预中心: 010-82951332
生命热线: 400-821-1215
希望 24 热线: 400-161-9995
如有紧急人身安全危险，请直接拨打 120 或 110。

重要提醒：
心理危机是可以被帮助的。拨打热线电话不需要预约，不需要提供真实姓名，热线工作人员会倾听你的困扰并提供即时的支持。你不需要独自面对这一切。"""
        crisis_path.write_text(crisis_content, encoding="utf-8")
        print(f"  ✅ 已创建 危机干预指引 → {crisis_path}")
    else:
        print("  ⏭️  危机干预文档已存在")

    print("  🎉 安全护栏文档创建完成!")
    return True


# ============================================================================
# 主入口
# ============================================================================
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     📦 PsyMind 数据集一键下载脚本                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  数据存放目录: {DATASETS_DIR}\n")

    results = {}

    # 按优先级依次下载
    results["PsyQA"] = download_psyqa()
    results["CPsyCounD"] = download_cpsycoun()
    results["mhGAP-IG"] = download_mhgap()
    results["OpenStax"] = download_openstax()
    results["Guardrails"] = create_guardrail_templates()

    # ── 汇总报告 ──
    print("\n" + "=" * 60)
    print("  📊 下载汇总")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {status}  {name}")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n  ⚠️  {len(failed)} 个数据集下载失败，请查看上方日志手动处理。")
    else:
        print(f"\n  🎉 全部数据集下载/创建成功!")

    print(f"\n  下一步: 运行预处理脚本")
    print(f"    python scripts/preprocess_datasets.py\n")


if __name__ == "__main__":
    main()
