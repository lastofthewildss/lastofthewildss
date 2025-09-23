(new_venv) PS D:\code\lastofthewildss\3223004474> python main_old.py texts/orig.txt texts/orig_0.8_add.txt result.txt
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\lamb\AppData\Local\Temp\jieba.cache
Loading model cost 0.742 seconds.
Prefix dict has been built successfully.
开始性能分析...
正在读取文件...
正在计算相似度...
查重完成！重复率: 89.40%
性能分析数据已保存到: profile_stats
正在启动snakeviz进行可视化分析...
snakeviz web server started on 127.0.0.1:8080; enter Ctrl-C to exit
http://127.0.0.1:8080/snakeviz/D%3A%5Ccode%5Clastofthewildss%5C3223004474%5Cprofile_stats

Bye!

程序被用户中断





(new_venv) PS D:\code\lastofthewildss\3223004474> python main.py texts/orig.txt texts/orig_0.8_add.txt result.txt
开始性能分析...
正在读取文件...
原文长度: 10511 字符
抄袭文本长度: 12274 字符
正在计算相似度...
查重完成！重复率: 84.43%
性能分析数据已保存到: profile_stats
启动snakeviz可视化...
snakeviz web server started on 127.0.0.1:8080; enter Ctrl-C to exit
http://127.0.0.1:8080/snakeviz/D%3A%5Ccode%5Clastofthewildss%5C3223004474%5Cprofile_stats

Bye!

程序被用户中断





(new_venv) PS D:\code\lastofthewildss\3223004474> python main.py texts/orig.txt texts/orig_0.8_del.txt result.txt
开始性能分析...
正在读取文件...
原文长度: 10511 字符
抄袭文本长度: 7964 字符
正在计算相似度...
查重完成！重复率: 84.02%

(new_venv) PS D:\code\lastofthewildss\3223004474> python main.py texts/orig.txt texts/orig_0.8_dis_1.txt result.txt
开始性能分析...
正在读取文件...
原文长度: 10511 字符
抄袭文本长度: 9702 字符
正在计算相似度...
查重完成！重复率: 90.82%





(new_venv) PS D:\code\lastofthewildss\3223004474> python main.py texts/orig.txt texts/orig_0.8_dis_10.txt result.txt
开始性能分析...
正在读取文件...
原文长度: 10511 字符
抄袭文本长度: 9797 字符
正在计算相似度...
查重完成！重复率: 85.94%

 PS D:\code\lastofthewildss\3223004474> python main.py texts/orig.txt texts/orig_0.8_dis_15.txt result.txt
开始性能分析...
正在读取文件...
原文长度: 10511 字符
抄袭文本长度: 9920 字符
正在计算相似度...
查重完成！重复率: 72.78%





(new_venv) PS D:\code\lastofthewildss\3223004474> coverage run -m unittest test_main.py
.........错误：无法解码文件 'invalid_encoding.txt'，请检查文件编码
.错误：文件 'nonexistent_file.txt' 不存在
.读取文件 'other_error.txt' 时发生错误：模拟其他异常
.错误：没有权限读取文件 'no_permission.txt'
.读取文件 'other_error.txt' 时发生错误：模拟其他异常
.写入文件 'no_permission.txt' 时发生错误：没有权限
.....正在读取文件...
原文长度: 22 字符
抄袭文本长度: 20 字符
正在计算相似度...
查重完成！重复率: 74.54%
.开始性能分析...
正在读取文件...
程序执行错误: 模拟异常
.开始性能分析...
性能分析数据已保存到: profile_stats
启动snakeviz可视化...
snakeviz web server started on 127.0.0.1:8080; enter Ctrl-C to exit
http://127.0.0.1:8080/snakeviz/D%3A%5Ccode%5Clastofthewildss%5C3223004474%5Cprofile_stats

Bye!

程序被用户中断



