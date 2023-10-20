"""
Auther: ncx@stu.ouc.edu.cn
Last updated: 2023/10/12
Requirements:
    python 3.9.16
"""


# 求去掉最大值和最小值的平均
def average_without_extremes(arr):

    min_val = min(arr)
    max_val = max(arr)
    arr.remove(min_val)
    arr.remove(max_val)
    total = sum(arr)
    average = total / len(arr)

    return average
