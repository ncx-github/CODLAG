"""
Auther: ncx@stu.ouc.edu.cn
Last Revision: 2023/10/20
Requirements:
    python 3.9.16
"""


# Find the average after removing the maximum and minimum values
def average_without_extremes(arr):

    min_val = min(arr)
    max_val = max(arr)
    arr.remove(min_val)
    arr.remove(max_val)
    total = sum(arr)
    average = total / len(arr)

    return average
