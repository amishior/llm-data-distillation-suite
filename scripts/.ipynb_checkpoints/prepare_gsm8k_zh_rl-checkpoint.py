import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split


def load_and_split_gsm8k_zh():
    """
    兼容两种情况：
    1）HF 只有一个 split=train，内部用 'split' 列区分 train/test
    2）HF 有 train/test 两个 split
    """
    ds_all = load_dataset("meta-math/GSM8K_zh")
    if "test" in ds_all:  # 显式 train/test
        train_raw = ds_all["train"]
        test_raw = ds_all["test"]
    else:
        raw = ds_all["train"]
        cols = raw.column_names
        if "split" in cols and len(set(raw["split"])) > 1:
            train_raw = raw.filter(lambda x: x["split"] == "train")
            test_raw = raw.filter(lambda x: x["split"] == "test")
        else:
            # 没有可靠的 split 列，就自己按 0.15 划分
            split = raw.train_test_split(test_size=0.15, seed=42)
            train_raw = split["train"]
            test_raw = split["test"]

    return DatasetDict({"train": train_raw, "test": test_raw})


def build_row(example, idx, split: str):
    """
    把 meta-math/GSM8K_zh 的一条样本，映射成 verl RL 期望的一行。
    已知列：
      - question
      - answer_only  # 纯数字答案
      - answer       # 英文解答 + #### 42
      - question_zh
      - answer_zh
      - split        # 有的版本只有 'train'
    参考: https://www.cnblogs.com/lx63blog/p/18906034
    """
    q_zh = example["question_zh"]
    a_zh = example["answer_zh"]
    ans_only = example.get("answer_only", None)

    # 1. data_source：一定要写 'openai/gsm8k'，这样 RewardManager 才会走内置 gsm8k 规则
    data_source = "openai/gsm8k"

    # 2. prompt：chat 格式，后面 tokenizer.apply_chat_template 会用到
    #    这里加一句中文指令，强制要求输出 "#### 最终答案"
    user_content = (
        q_zh
        + "\n\n请一步一步推理，并在最后一行用“#### 最终答案：<数字>”的格式给出最终答案。"
    )
    prompt = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    # 3. ability：照抄官方
    ability = "math"

    # 4. reward_model：用 answer_only 作为 ground truth
    #    如果没有 answer_only，可以从英文 answer / 中文 answer_zh 里自己 parse，这里简单用 answer_only。
    reward_model = {
        "style": "rule",
        "ground_truth": ans_only if ans_only is not None else a_zh,
    }

    # 5. extra_info：保留原始中英题目与答案，方便 debug / 之后做别的用
    extra_info = {
        "question_en": example.get("question", None),
        "answer_en": example.get("answer", None),
        "question_zh": q_zh,
        "answer_zh": a_zh,
        "answer_only": ans_only,
        "orig_split": example.get("split", split),
        "idx": int(idx),
    }

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def main():
    ds = load_and_split_gsm8k_zh()
    rows_train = []
    rows_test = []

    for idx, ex in enumerate(ds["train"]):
        rows_train.append(build_row(ex, idx, "train"))

    for idx, ex in enumerate(ds["test"]):
        rows_test.append(build_row(ex, idx, "test"))

    df_train = pd.DataFrame(rows_train)
    df_test = pd.DataFrame(rows_test)

    print("train rows:", len(df_train))
    print("test rows:", len(df_test))
    print("columns:", df_train.columns.tolist())

    save_dir = Path("/home/amishor/data/gsm8k_zh_rl")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_path = save_dir / "train.parquet"
    test_path = save_dir / "test.parquet"

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    print("Saved:")
    print("  train ->", train_path)
    print("  test  ->", test_path)


if __name__ == "__main__":
    main()
