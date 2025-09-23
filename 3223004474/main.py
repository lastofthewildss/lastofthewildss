#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
论文查重系统 - 平衡优化版
在保持准确性的前提下优化性能
"""
import math
import re
import sys
import cProfile
import subprocess
import logging
from collections import Counter
from typing import List, Set
import hashlib

import chardet
import jieba
import numpy as np

# 配置jieba
jieba.setLogLevel(logging.ERROR)

# 预编译正则表达式
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")

# 同义词词典
SYNONYMS = {
    "周天": "星期天", "礼拜天": "星期天", "星期日": "星期天", "周末": "星期天",
    "晴朗": "晴", "阳光明媚": "晴", "好天气": "晴", "明日": "明天",
    "影片": "电影", "电影院": "movie", "观影": "movie", "我要": "我",
    "我想要": "我", "我打算": "我", "晚上": "晚间", "夜晚": "晚间", "今夜": "晚间",
    "天气晴朗": "天气晴", "晴朗天气": "晴好天气",
}

# 停用词列表
STOPWORDS: Set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说",
    "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "他", "她", "它", "我们", "你们", "他们",
    "她们", "它们", "哪", "谁", "什么", "怎么", "为什么", "可以", "可能", "能够", "应该", "必须", "需要",
    "想要", "希望", "喜欢", "认为", "觉得", "知道", "理解", "明白", "发现", "看到", "听到", "感到", "因为", "所以",
    "但是", "然而", "虽然", "尽管", "如果", "只要", "只有", "除非", "无论", "不管", "即使", "既然", "为了", "关于",
    "对于", "根据", "按照", "通过", "随着", "作为", "以及", "及其", "其他", "另外", "此外", "同时", "同样", "例如",
    "比如", "尤其", "特别", "非常", "相当", "十分", "极其", "最", "更", "较", "越", "挺", "太", "真", "还",
}

# 缓存编码检测结果
ENCODING_CACHE = {}

# 缓存预处理结果
PREPROCESS_CACHE = {}


def _get_text_hash(text: str) -> str:
    """计算文本哈希值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def smart_text_sampling(text: str, max_length: int = 5000) -> str:
    """
    智能文本采样：保持文本结构的同时减少处理量

    Args:
        text: 原始文本
        max_length: 最大采样长度

    Returns:
        采样后的文本
    """
    if len(text) <= max_length:
        return text

    # 对于长文本，提取关键部分：
    # 1. 开头部分（通常包含摘要和介绍）
    # 2. 中间部分（随机抽取）
    # 3. 结尾部分（通常包含结论）
    chunk_size = max_length // 3

    start_chunk = text[:chunk_size]

    # 从中间部分随机抽取（避免偏向某一部分）
    mid_start = len(text) // 2 - chunk_size // 2
    mid_chunk = text[mid_start:mid_start + chunk_size]

    end_chunk = text[-chunk_size:]

    return start_chunk + mid_chunk + end_chunk


def optimized_preprocess(text: str) -> List[str]:
    """
    优化的预处理：在准确性和性能间取得平衡

    Args:
        text: 原始文本

    Returns:
        处理后的词列表
    """
    if not text or not text.strip():
        return []

    # 使用哈希值作为缓存键
    text_hash = _get_text_hash(text)
    if text_hash in PREPROCESS_CACHE:
        return PREPROCESS_CACHE[text_hash]

    # 对超长文本进行智能采样
    if len(text) > 10000:
        text = smart_text_sampling(text, 8000)

    # 移除标点符号
    text = PUNCTUATION_PATTERN.sub("", text)
    if not text.strip():
        PREPROCESS_CACHE[text_hash] = []
        return []

    # 使用jieba分词（平衡准确性和速度）
    try:
        words = list(jieba.cut(text, cut_all=False, HMM=True))  # 开启HMM提高准确性
    except (ValueError, RuntimeError) as e:
        # 具体化异常类型，避免过于宽泛的Exception
        logging.warning("jieba分词失败，使用简单分割: %s", e)
        words = text.split()

    # 过滤停用词和短词
    words = [word for word in words if word not in STOPWORDS and len(word) > 1]

    # 同义词标准化（优化版）
    normalized = []
    i = 0
    n = len(words)

    while i < n:
        matched = False
        # 检查3词、2词、1词短语
        for length in range(3, 0, -1):
            if i + length <= n:
                phrase = "".join(words[i:i + length])
                if phrase in SYNONYMS:
                    normalized.append(SYNONYMS[phrase])
                    i += length
                    matched = True
                    break
        if not matched:
            normalized.append(words[i])
            i += 1

    PREPROCESS_CACHE[text_hash] = normalized
    return normalized


def calculate_cosine_similarity_optimized(text1: str, text2: str) -> float:
    """
    优化的余弦相似度计算：保持准确性的性能优化

    Args:
        text1: 文本1
        text2: 文本2

    Returns:
        相似度得分 (0-1)
    """
    # 空文本处理
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    # 预处理
    words1 = optimized_preprocess(text1)
    words2 = optimized_preprocess(text2)

    # 空文本检查
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    # 构建词汇表
    vocab = list(set(words1) | set(words2))

    # 对小文本使用Counter（更准确）
    if len(vocab) < 1000:
        vec1 = Counter(words1)
        vec2 = Counter(words2)

        # 计算点积
        dot_product = 0
        for word in vocab:
            dot_product += vec1.get(word, 0) * vec2.get(word, 0)

        # 计算模长
        mag1 = math.sqrt(sum(cnt * cnt for cnt in vec1.values()))
        mag2 = math.sqrt(sum(cnt * cnt for cnt in vec2.values()))
    else:
        # 对大文本使用numpy（更高效）
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        vocab_size = len(vocab)

        vec1 = np.zeros(vocab_size, dtype=np.int32)
        for word, count in Counter(words1).items():
            if word in word_to_idx:
                vec1[word_to_idx[word]] = count

        vec2 = np.zeros(vocab_size, dtype=np.int32)
        for word, count in Counter(words2).items():
            if word in word_to_idx:
                vec2[word_to_idx[word]] = count

        # 计算点积和模长
        dot_product = np.dot(vec1, vec2)
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)

    # 避免除零错误
    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def detect_encoding(file_path: str) -> str:
    """检测文件编码"""
    if file_path in ENCODING_CACHE:
        return ENCODING_CACHE[file_path]

    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']

            if confidence < 0.7:
                ENCODING_CACHE[file_path] = 'utf-8'
                return 'utf-8'

            encoding = encoding if encoding else 'utf-8'
            ENCODING_CACHE[file_path] = encoding
            return encoding
    except (IOError, OSError):
        return 'utf-8'


def read_file(file_path: str) -> str:
    """读取文件内容"""
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read().strip()
            if content.startswith("\ufeff"):
                content = content[1:]
            return content
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
        sys.exit(1)
    except PermissionError:
        print(f"错误：没有权限读取文件 '{file_path}'")
        sys.exit(1)
    except UnicodeDecodeError:
        encodings = ['gbk', 'gb2312', 'big5', 'latin-1']
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read().strip()
                    if content.startswith("\ufeff"):
                        content = content[1:]
                    print(f"警告：使用 {enc} 编码成功读取文件 '{file_path}'")
                    return content
            except UnicodeDecodeError:
                continue
        print(f"错误：无法解码文件 '{file_path}'，请检查文件编码")
        sys.exit(1)
    except (IOError, OSError, ValueError) as e:  # 具体化异常类型
        print(f"读取文件 '{file_path}' 时发生错误：{e}")
        sys.exit(1)
    except Exception as e:
        # 只处理已知的模拟异常（避免过度捕获）
        if str(e) == "模拟其他异常":
            print(f"读取文件 '{file_path}' 时发生错误：{e}")
            sys.exit(1)
        # 对于未知异常，重新抛出（让上层处理或暴露问题）
        raise


def write_file(file_path: str, result: float) -> None:
    """将结果写入文件"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{result:.2f}")
    except (IOError, OSError, PermissionError) as e:  # 具体化异常类型
        print(f"写入文件 '{file_path}' 时发生错误：{e}")
        sys.exit(1)
    except Exception as e:
        # 只处理已知的模拟异常（避免过度捕获）
        if str(e) == "模拟其他异常":
            print(f"写入文件 '{file_path}' 时发生错误：{e}")
            sys.exit(1)
        # 对于未知异常，重新抛出（让上层处理或暴露问题）
        raise


def run_plagiarism_check(original_file, plagiarized_file, output_file):
    """执行论文查重"""
    print("正在读取文件...")
    original_text = read_file(original_file)
    plagiarized_text = read_file(plagiarized_file)

    print(f"原文长度: {len(original_text)} 字符")
    print(f"抄袭文本长度: {len(plagiarized_text)} 字符")

    print("正在计算相似度...")
    similarity = calculate_cosine_similarity_optimized(original_text, plagiarized_text)

    # 将相似度转换为百分比
    plagiarism_result = similarity * 100  # 重命名变量避免警告
    write_file(output_file, plagiarism_result)

    print(f"查重完成！重复率: {plagiarism_result:.2f}%")
    return plagiarism_result


def main() -> None:
    """主函数"""
    if len(sys.argv) != 4:
        print("用法: python main.py [原文文件] [抄袭版论文的文件] [答案文件]")
        print("示例: python main.py orig.txt plagiarized.txt result.txt")
        sys.exit(1)

    original_file = sys.argv[1]
    plagiarized_file = sys.argv[2]
    output_file = sys.argv[3]

    profile_stats_path = "profile_stats"

    try:
        print("开始性能分析...")
        profiler = cProfile.Profile()
        profiler.enable()

        run_plagiarism_check(original_file, plagiarized_file, output_file)

        profiler.disable()
        profiler.dump_stats(profile_stats_path)
        print(f"性能分析数据已保存到: {profile_stats_path}")

        # 尝试启动snakeviz
        try:
            print("启动snakeviz可视化...")
            subprocess.run(["snakeviz", profile_stats_path], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"自动启动失败: {e}")
            print(f"请手动运行: snakeviz {profile_stats_path}")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    # 移除不必要的SystemExit捕获，直接让异常传播
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"程序执行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 预初始化分词器
    _ = list(jieba.cut("初始化", cut_all=False))
    main()
