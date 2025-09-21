#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
论文查重系统单元测试
测试main.py中的所有功能
"""

import unittest
import os
import tempfile
from unittest.mock import patch
from main import read_file, write_file, preprocess, calculate_cosine_similarity, main

class TestPaperCheck(unittest.TestCase):

    def setUp(self):
        """在每个测试方法前执行，用于设置测试环境"""
        # 创建测试用的临时文件
        with open("test_orig.txt", "w", encoding="utf-8") as f:
            f.write("今天是星期天，天气晴，今天晚上我要去看电影。")

        with open("test_plag.txt", "w", encoding="utf-8") as f:
            f.write("今天是周天，天气晴朗，我晚上要去看电影。")

        with open("empty_file.txt", "w", encoding="utf-8") as f:
            f.write("")  # 创建空文件

        with open("test_numbers.txt", "w", encoding="utf-8") as f:
            f.write("这是一个测试123")

        with open("test_punctuation.txt", "w", encoding="utf-8") as f:
            f.write("这是一个测试，包含标点！")

    def tearDown(self):
        """在每个测试方法后执行，用于清理测试环境"""
        # 删除测试用的临时文件
        test_files = [
            "test_orig.txt",
            "test_plag.txt",
            "test_output.txt",
            "empty_file.txt",
            "test_numbers.txt",
            "test_punctuation.txt",
        ]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)

    def test_read_file_normal(self):
        """测试正常文件读取功能"""
        content = read_file("test_orig.txt")
        self.assertEqual(content, "今天是星期天，天气晴，今天晚上我要去看电影。")

    def test_read_file_nonexistent(self):
        """测试读取不存在文件时的异常处理"""
        with self.assertRaises(SystemExit):
            read_file("nonexistent_file.txt")

    @patch("builtins.open", side_effect=PermissionError("没有权限"))
    def test_read_file_permission_error(self, mock_open):
        """测试读取无权限文件时的异常处理"""
        with self.assertRaises(SystemExit):
            read_file("no_permission.txt")

    @patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "Invalid UTF-8"))
    def test_read_file_encoding_error(self, mock_open):
        """测试读取编码错误文件时的异常处理"""
        with self.assertRaises(SystemExit):
            read_file("invalid_encoding.txt")

    @patch("builtins.open", side_effect=Exception("模拟其他异常"))
    def test_read_file_other_exception(self, mock_open):
        """测试读取文件时遇到其他异常的情况"""
        with self.assertRaises(SystemExit):
            read_file("other_error.txt")

    def test_write_file_normal(self):
        """测试文件写入功能"""
        write_file("test_output.txt", 75.50)
        with open("test_output.txt", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "75.50")

    @patch("builtins.open", side_effect=PermissionError("没有权限"))
    def test_write_file_permission_error(self, mock_open):
        """测试写入无权限文件时的异常处理"""
        with self.assertRaises(SystemExit):
            write_file("no_permission.txt", 75.50)

    @patch("builtins.open", side_effect=Exception("模拟其他异常"))
    def test_write_file_other_exception(self, mock_open):
        """测试写入文件时遇到其他异常的情况"""
        with self.assertRaises(SystemExit):
            write_file("other_error.txt", 75.50)

    def test_preprocess_normal(self):
        """测试文本预处理功能"""
        text = "今天是星期天，天气晴，今天晚上我要去看电影。"
        result = preprocess(text)
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
        result = preprocess(text)
        self.assertEqual(result, [])

    def test_preprocess_with_numbers(self):
        """测试包含数字的文本预处理"""
        text = "这是一个测试123"
        result = preprocess(text)
        # 数字应该被保留
        self.assertIn("123", result)

    def test_preprocess_with_punctuation(self):
        """测试包含标点符号的文本预处理"""
        text = "这是一个测试，包含标点！"
        result = preprocess(text)
        # 标点符号应该被移除
        self.assertNotIn("，", result)
        self.assertNotIn("！", result)
        # 应该包含处理后的词语
        self.assertIn("测试", result)
        self.assertIn("包含", result)
        self.assertIn("标点", result)

    def test_cosine_similarity_identical(self):
        """测试相同文本的相似度计算"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_cosine_similarity_different(self):
        """测试完全不同文本的相似度计算"""
        text1 = "今天是星期天"
        text2 = "明天是星期一"
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertLess(similarity, 0.5)

    def test_cosine_similarity_partial(self):
        """测试部分相似文本的相似度计算"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是周天，天气晴朗，我晚上要去看电影。"
        similarity = calculate_cosine_similarity(text1, text2)
        # 由于同义词处理，预期相似度应该较高
        self.assertGreater(similarity, 0.7)
        self.assertLessEqual(similarity, 1.0)

    def test_cosine_similarity_empty(self):
        """测试空文本的相似度计算"""
        text1 = ""
        text2 = "今天是星期天"
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertEqual(similarity, 0.0)

    def test_cosine_similarity_both_empty(self):
        """测试两个空文本的相似度计算"""
        text1 = ""
        text2 = ""
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertEqual(similarity, 1.0)

    def test_cosine_similarity_with_numbers(self):
        """测试包含数字的文本相似度计算"""
        text1 = "这是一个测试123"
        text2 = "这是一个测试123"
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_cosine_similarity_with_punctuation(self):
        """测试包含标点符号的文本相似度计算"""
        text1 = "这是一个测试，包含标点！"
        text2 = "这是一个测试，包含标点！"
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_cosine_similarity_long_text(self):
        """测试长文本的相似度计算"""
        text1 = "今天天气很好，我们一起去公园玩。公园里有很多人，有的在散步，有的在跑步，还有的在骑自行车。我们看到了一只可爱的小狗，它在追着球跑。"
        text2 = "今天天气很好，我们一起去公园玩。公园里有很多人，有的在散步，有的在跑步，还有的在骑自行车。我们看到了一只可爱的小狗，它在追着球跑。"
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_cosine_similarity_english(self):
        """测试英文文本的相似度计算"""
        text1 = "This is a test for english text."
        text2 = "This is a test for english text."
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_cosine_similarity_mixed_language(self):
        """测试中英文混合文本的相似度计算"""
        text1 = "这是一个test，包含中英文mixed。"
        text2 = "这是一个test，包含中英文mixed。"
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    @patch("sys.argv", ["main.py", "test_orig.txt", "test_plag.txt", "test_output.txt"])
    @patch("main.read_file")
    @patch("main.calculate_cosine_similarity")
    @patch("main.write_file")
    def test_main_normal(self, mock_write, mock_calc, mock_read):
        """测试主函数正常流程"""
        # 设置mock返回值
        mock_read.side_effect = ["原文内容", "抄袭内容"]
        mock_calc.return_value = 0.85

        # 调用主函数
        main()

        # 验证函数调用
        self.assertEqual(mock_read.call_count, 2)
        mock_calc.assert_called_once_with("原文内容", "抄袭内容")
        mock_write.assert_called_once_with("test_output.txt", 85.0)

    @patch("sys.argv", ["main.py"])
    def test_main_insufficient_arguments(self):
        """测试参数不足时的异常处理"""
        with self.assertRaises(SystemExit):
            main()

    @patch("sys.argv", ["main.py", "file1.txt", "file2.txt", "output.txt"])
    @patch("main.read_file", side_effect=Exception("模拟异常"))
    def test_main_exception_handling(self, mock_read):
        """测试主函数异常处理"""
        with self.assertRaises(SystemExit):
            main()

if __name__ == "__main__":
    unittest.main()