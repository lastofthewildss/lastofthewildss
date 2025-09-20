import sys
import jieba
import numpy as np
from collections import Counter


def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"读取文件时出错：{e}")
        sys.exit(1)


def process_text(text):
    """处理文本：分词并去除停用词"""
    # 这里可以添加更多文本预处理步骤
    words = jieba.cut(text)
    return [word for word in words if word.strip()]


def calculate_similarity(orig_words, copy_words):
    """计算余弦相似度"""
    # 获取所有唯一词
    all_words = set(orig_words + copy_words)

    # 创建词频向量
    orig_vector = [orig_words.count(word) for word in all_words]
    copy_vector = [copy_words.count(word) for word in all_words]

    # 计算余弦相似度
    dot_product = np.dot(orig_vector, copy_vector)
    norm_orig = np.linalg.norm(orig_vector)
    norm_copy = np.linalg.norm(copy_vector)

    if norm_orig == 0 or norm_copy == 0:
        return 0.0

    return dot_product / (norm_orig * norm_copy)


def main():
    if len(sys.argv) != 4:
        print("用法: python main.py [原文文件] [抄袭版论文] [答案文件]")
        sys.exit(1)

    orig_path, copy_path, answer_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # 读取文件
    orig_text = read_file(orig_path)
    copy_text = read_file(copy_path)

    # 处理文本
    orig_words = process_text(orig_text)
    copy_words = process_text(copy_text)

    # 计算相似度
    similarity = calculate_similarity(orig_words, copy_words)

    # 写入结果
    try:
        with open(answer_path, 'w', encoding='utf-8') as f:
            f.write(f"{similarity:.2f}")
        print(f"查重完成，结果已保存到 {answer_path}")
    except Exception as e:
        print(f"写入结果时出错：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()