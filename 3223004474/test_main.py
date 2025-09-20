import unittest
import os
import tempfile
from main import read_file, process_text, calculate_similarity


class TestPlagiarismCheck(unittest.TestCase):

    def setUp(self):
        # 创建临时测试文件
        self.test_dir = tempfile.mkdtemp()
        self.orig_path = os.path.join(self.test_dir, "orig.txt")
        self.copy_path = os.path.join(self.test_dir, "copy.txt")
        self.answer_path = os.path.join(self.test_dir, "answer.txt")

        with open(self.orig_path, 'w', encoding='utf-8') as f:
            f.write("今天是星期天，天气晴，今天晚上我要去看电影。")

        with open(self.copy_path, 'w', encoding='utf-8') as f:
            f.write("今天是周天，天气晴朗，我晚上要去看电影。")

    def test_read_file(self):
        content = read_file(self.orig_path)
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def test_process_text(self):
        text = "今天是星期天，天气晴"
        words = process_text(text)
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 0)

    def test_calculate_similarity(self):
        orig_words = ["今天", "是", "星期天", "天气", "晴"]
        copy_words = ["今天", "是", "周天", "天气", "晴朗"]
        similarity = calculate_similarity(orig_words, copy_words)
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)

    def test_identical_texts(self):
        orig_words = ["今天", "是", "星期天"]
        copy_words = ["今天", "是", "星期天"]
        similarity = calculate_similarity(orig_words, copy_words)
        self.assertEqual(similarity, 1.0)

    def test_completely_different_texts(self):
        orig_words = ["今天", "是", "星期天"]
        copy_words = ["明天", "要", "下雨"]
        similarity = calculate_similarity(orig_words, copy_words)
        self.assertEqual(similarity, 0.0)

    def test_empty_text(self):
        similarity = calculate_similarity([], [])
        self.assertEqual(similarity, 0.0)

    # 可以继续添加更多测试用例...


if __name__ == '__main__':
    unittest.main()