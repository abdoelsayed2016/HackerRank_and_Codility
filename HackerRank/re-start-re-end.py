"""
start() & end()
These expressions return the indices of the start and end of the substring matched by the group.

Code

>>> import re
>>> m = re.search(r'\d+','1234')
>>> m.end()
4
>>> m.start()
0
Task
You are given a string .
Your task is to find the indices of the start and end of string  in .

Input Format

The first line contains the string .
The second line contains the string .

Constraints



Output Format

Print the tuple in this format: (start _index, end _index).
If no match is found, print (-1, -1).

Sample Input

aaadaa
aa
Sample Output

(0, 1)
(1, 2)
(4, 5)
"""

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

text = input()
s = input()
index = 0
c = []
while index < len(text):
    index = text.find(s, index)
    if index == -1:
        break

    c.append((index, index + len(s) - 1))
    index += 1  # +2 because len('ll') == 2

if c:
    for x in c:
        print(x)
else:
    print("(-1, -1)")



