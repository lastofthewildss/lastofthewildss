#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
论文查重系统单元测试 - 完整覆盖率版本
测试main.py中的所有功能
"""

import os
import unittest
from unittest.mock import patch
import tempfile
import shutil
from main import main

from main import (
    read_file, write_file, optimized_preprocess,
    calculate_cosine_similarity_optimized, smart_text_sampling,
    detect_encoding, run_plagiarism_check, _get_text_hash
)


class TestFileOperations(unittest.TestCase):
    """测试文件操作功能的类"""

    def setUp(self):
        """在每个测试方法前执行，用于设置测试环境"""
        self.test_dir = tempfile.mkdtemp()

        # 创建测试文件
        self.orig_file = os.path.join(self.test_dir, "test_orig.txt")
        with open(self.orig_file, "w", encoding="utf-8") as f:
            f.write("今天是星期天，天气晴，今天晚上我要去看电影。")

        self.output_file = os.path.join(self.test_dir, "test_output.txt")

    def tearDown(self):
        """在每个测试方法后执行，用于清理测试环境"""
        shutil.rmtree(self.test_dir)

    def test_read_file_normal(self):
        """测试正常文件读取功能"""
        content = read_file(self.orig_file)
        self.assertEqual(content, "今天是星期天，天气晴，今天晚上我要去看电影。")

    def test_write_file_normal(self):
        """测试文件写入功能"""
        write_file(self.output_file, 75.50)
        with open(self.output_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "75.50")

    def test_detect_encoding_high_confidence(self):
        """测试高置信度编码检测"""
        # 创建UTF-8编码文件
        test_file = os.path.join(self.test_dir, "test_utf8.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("测试UTF-8编码")

        encoding = detect_encoding(test_file)
        self.assertEqual(encoding, "utf-8")

    def test_detect_encoding_low_confidence(self):
        """测试低置信度编码检测"""
        # 创建短文本文件（置信度可能较低）
        test_file = os.path.join(self.test_dir, "test_short.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ab")  # 很短的内容

        encoding = detect_encoding(test_file)
        # 修复：接受ascii编码，因为短文本可能被识别为ascii
        self.assertIn(encoding, ['utf-8', 'ascii'])


class TestTextSampling(unittest.TestCase):
    """测试文本采样功能"""

    def test_smart_sampling_short_text(self):
        """测试短文本采样（应返回原文本）"""
        short_text = "这是一个短文本"
        result = smart_text_sampling(short_text, max_length=100)
        self.assertEqual(result, short_text)

    def test_smart_sampling_long_text(self):
        """测试长文本采样"""
        # 创建长文本（超过最大长度）
        long_text = "开头部分" + "中间内容" * 1000 + "结尾部分"
        result = smart_text_sampling(long_text, max_length=100)

        # 检查采样结果包含关键部分
        self.assertIn("开头部分", result)
        self.assertIn("结尾部分", result)
        self.assertLessEqual(len(result), 100)

    def test_smart_sampling_exact_length(self):
        """测试恰好等于最大长度的文本"""
        exact_text = "a" * 5000
        result = smart_text_sampling(exact_text, max_length=5000)
        self.assertEqual(result, exact_text)


class TestPreprocessing(unittest.TestCase):
    """测试文本预处理功能的类"""

    def test_preprocess_normal(self):
        """测试正常文本预处理功能"""
        text = "今天是星期天，天气晴，今天晚上我要去看电影。"
        result = optimized_preprocess(text)

        # 检查是否返回列表
        self.assertIsInstance(result, list)
        # 检查同义词处理
        self.assertIn("星期天", result)
        # 检查停用词过滤
        self.assertNotIn("的", result)
        self.assertNotIn("了", result)

    def test_preprocess_empty(self):
        """测试空文本预处理"""
        text = ""
        result = optimized_preprocess(text)
        self.assertEqual(result, [])

    def test_preprocess_only_punctuation(self):
        """测试只有标点符号的文本"""
        text = "！@#￥%……&*（）"
        result = optimized_preprocess(text)
        self.assertEqual(result, [])

    def test_preprocess_with_numbers(self):
        """测试包含数字的文本预处理"""
        text = "这是一个测试123"
        result = optimized_preprocess(text)
        # 数字应该被保留
        self.assertIn("123", result)

    def test_preprocess_with_punctuation(self):
        """测试包含标点符号的文本预处理"""
        text = "这是一个测试，包含标点！"
        result = optimized_preprocess(text)
        # 标点符号应该被移除
        self.assertNotIn("，", result)
        self.assertNotIn("！", result)
        # 应该包含处理后的词语
        self.assertIn("测试", result)
        self.assertIn("包含", result)
        self.assertIn("标点", result)

    def test_preprocess_synonym_replacement(self):
        """测试同义词替换功能"""
        text = "今天是周天，天气晴朗，我晚上要去看电影。"
        result = optimized_preprocess(text)
        # 检查同义词是否被正确替换
        self.assertIn("星期天", result)
        # 修复：检查"天气晴"而不是单独的"晴"，因为"天气晴朗"被替换为"天气晴"
        self.assertIn("天气晴", result)
        self.assertIn("晚间", result)

    def test_preprocess_long_text_sampling(self):
        """测试长文本的智能采样预处理"""
        # 创建超长文本
        long_text = "开头" + "重复内容" * 5000 + "结尾"
        result = optimized_preprocess(long_text)
        # 应该返回非空结果
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)


class TestCosineSimilarity(unittest.TestCase):
    """测试余弦相似度计算功能的类"""

    def test_cosine_similarity_identical(self):
        """测试相同文本的相似度计算"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        similarity = calculate_cosine_similarity_optimized(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_cosine_similarity_different(self):
        """测试完全不同文本的相似度计算"""
        text1 = "今天是星期天"
        text2 = "明天是星期一"
        similarity = calculate_cosine_similarity_optimized(text1, text2)
        self.assertLess(similarity, 0.5)

    def test_cosine_similarity_partial(self):
        """测试部分相似文本的相似度计算"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是周天，天气晴朗，我晚上要去看电影。"
        similarity = calculate_cosine_similarity_optimized(text1, text2)
        # 由于同义词处理，预期相似度应该较高
        self.assertGreater(similarity, 0.7)
        self.assertLessEqual(similarity, 1.0)

    def test_cosine_similarity_empty_first(self):
        """测试第一个文本为空的情况"""
        text1 = ""
        text2 = "今天是星期天"
        similarity = calculate_cosine_similarity_optimized(text1, text2)
        self.assertEqual(similarity, 0.0)

    def test_cosine_similarity_empty_second(self):
        """测试第二个文本为空的情况"""
        text1 = "今天是星期天"
        text2 = ""
        similarity = calculate_cosine_similarity_optimized(text1, text2)
        self.assertEqual(similarity, 0.0)

    def test_cosine_similarity_both_empty(self):
        """测试两个空文本的相似度计算"""
        text1 = ""
        text2 = ""
        similarity = calculate_cosine_similarity_optimized(text1, text2)
        self.assertEqual(similarity, 1.0)

    def test_cosine_similarity_after_preprocess_empty(self):
        """测试预处理后变为空文本的情况"""
        text1 = "！@#￥%"
        text2 = "！@#￥%"
        similarity = calculate_cosine_similarity_optimized(text1, text2)
        self.assertEqual(similarity, 1.0)

    def test_cosine_similarity_small_vocab(self):
        """测试小词汇表的计算路径"""
        text1 = "苹果 香蕉 橙子"
        text2 = "苹果 香蕉 梨"
        similarity = calculate_cosine_similarity_optimized(text1, text2)
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)

    def test_cosine_similarity_large_vocab(self):
        """测试大词汇表的计算路径（触发numpy路径）"""
        # 创建包含大量不同词汇的文本
        text1_words = [f"词{i}" for i in range(1500)]  # 确保词汇量超过1000
        text2_words = [f"词{i}" for i in range(1000, 2000)]

        text1 = " ".join(text1_words)
        text2 = " ".join(text2_words)

        similarity = calculate_cosine_similarity_optimized(text1, text2)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)


class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""

    def test_get_text_hash(self):
        """测试文本哈希函数"""
        text1 = "相同文本"
        text2 = "相同文本"
        text3 = "不同文本"

        hash1 = _get_text_hash(text1)
        hash2 = _get_text_hash(text2)
        hash3 = _get_text_hash(text3)

        self.assertEqual(hash1, hash2)  # 相同文本应该产生相同哈希
        self.assertNotEqual(hash1, hash3)  # 不同文本应该产生不同哈希


class TestErrorHandling(unittest.TestCase):
    """测试错误处理功能的类"""

    def test_read_file_nonexistent(self):
        """测试读取不存在文件时的异常处理"""
        with self.assertRaises(SystemExit):
            read_file("nonexistent_file.txt")

    @patch("builtins.open", side_effect=PermissionError("没有权限"))
    def test_read_file_permission_error(self, _):
        """测试读取无权限文件时的异常处理"""
        with self.assertRaises(SystemExit):
            read_file("no_permission.txt")

    @patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "Invalid UTF-8"))
    @patch("main.detect_encoding", return_value="utf-8")
    def test_read_file_encoding_error(self, _, __):
        """测试读取编码错误文件时的异常处理"""
        with self.assertRaises(SystemExit):
            read_file("invalid_encoding.txt")

    @patch("builtins.open", side_effect=Exception("模拟其他异常"))
    def test_read_file_other_exception(self, _):
        """测试读取文件时遇到其他异常的情况"""
        with self.assertRaises(SystemExit):
            read_file("other_error.txt")

    @patch("builtins.open", side_effect=PermissionError("没有权限"))
    def test_write_file_permission_error(self, _):
        """测试写入无权限文件时的异常处理"""
        with self.assertRaises(SystemExit):
            write_file("no_permission.txt", 75.50)

    @patch("builtins.open", side_effect=Exception("模拟其他异常"))
    def test_write_file_other_exception(self, _):
        """测试写入文件时遇到其他异常的情况"""
        with self.assertRaises(SystemExit):
            write_file("other_error.txt", 75.50)


class TestIntegration(unittest.TestCase):
    """测试集成功能"""

    def setUp(self):
        """设置集成测试环境"""
        self.test_dir = tempfile.mkdtemp()

        # 创建测试文件
        self.orig_file = os.path.join(self.test_dir, "orig.txt")
        self.plag_file = os.path.join(self.test_dir, "plagiarized.txt")
        self.output_file = os.path.join(self.test_dir, "result.txt")

        with open(self.orig_file, "w", encoding="utf-8") as f:
            f.write("今天是星期天，天气晴，今天晚上我要去看电影。")

        with open(self.plag_file, "w", encoding="utf-8") as f:
            f.write("今天是周天，天气晴朗，我晚上要去看电影。")

    def tearDown(self):
        """清理集成测试环境"""
        shutil.rmtree(self.test_dir)

    def test_run_plagiarism_check_normal(self):
        """测试完整的查重流程"""
        result = run_plagiarism_check(self.orig_file, self.plag_file, self.output_file)

        # 检查结果在合理范围内
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 100.0)

        # 检查输出文件是否正确生成
        self.assertTrue(os.path.exists(self.output_file))
        with open(self.output_file, "r", encoding="utf-8") as f:
            content = f.read()
            # 应该是两位小数的浮点数
            self.assertRegex(content, r"^\d+\.\d{2}$")




class TestMainFunction(unittest.TestCase):
    """测试主函数"""

    @patch("sys.argv", ["main.py", "orig.txt", "plagiarized.txt", "result.txt"])
    @patch("main.run_plagiarism_check")
    def test_main_function(self, mock_run):
        """测试主函数"""
        mock_run.return_value = 85.0

        # 这里应该正常执行，不抛出异常
        try:
            main()
        except SystemExit:
            pass  # main函数会调用sys.exit()

    @patch("sys.argv", ["main.py"])  # 参数不足
    def test_main_insufficient_arguments(self):
        """测试参数不足时的异常处理"""
        with self.assertRaises(SystemExit):
            main()

    @patch("sys.argv", ["main.py", "file1.txt", "file2.txt", "output.txt"])
    @patch("main.read_file", side_effect=Exception("模拟异常"))
    def test_main_exception_handling(self, _):
        """测试主函数异常处理"""
        with self.assertRaises(SystemExit):
            main()


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
