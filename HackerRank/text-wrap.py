"""
You are given a string  and width .
Your task is to wrap the string into a paragraph of width .

Input Format

The first line contains a string, .
The second line contains the width, .

Constraints

Output Format

Print the text wrapped paragraph.

Sample Input 0

ABCDEFGHIJKLIMNOQRSTUVWXYZ
4
Sample Output 0

ABCD
EFGH
IJKL
IMNO
QRST
UVWX
YZ
"""

import math
def wrap(string, max_width):
    """num=math.ceil(len(string)/max_width)
    s=''
    index=0
    end=max_width
    for i in range(num):
        s+=string[index:end]+"\n"
        index+=max_width
        end+=max_width"""
    return textwrap.fill(string, max_width)