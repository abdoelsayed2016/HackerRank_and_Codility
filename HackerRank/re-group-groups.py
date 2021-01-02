"""
group()
A group() expression returns one or more subgroups of the match.
Code

>>> import re
>>> m = re.match(r'(\w+)@(\w+)\.(\w+)','username@hackerrank.com')
>>> m.group(0)       # The entire match
'username@hackerrank.com'
>>> m.group(1)       # The first parenthesized subgroup.
'username'
>>> m.group(2)       # The second parenthesized subgroup.
'hackerrank'
>>> m.group(3)       # The third parenthesized subgroup.
'com'
>>> m.group(1,2,3)   # Multiple arguments give us a tuple.
('username', 'hackerrank', 'com')
groups()
A groups() expression returns a tuple containing all the subgroups of the match.
Code

>>> import re
>>> m = re.match(r'(\w+)@(\w+)\.(\w+)','username@hackerrank.com')
>>> m.groups()
('username', 'hackerrank', 'com')
groupdict()
A groupdict() expression returns a dictionary containing all the named subgroups of the match, keyed by the subgroup name.
Code

>>> m = re.match(r'(?P<user>\w+)@(?P<website>\w+)\.(?P<extension>\w+)','myname@hackerrank.com')
>>> m.groupdict()
{'website': 'hackerrank', 'user': 'myname', 'extension': 'com'}
Task

You are given a string .
Your task is to find the first occurrence of an alphanumeric character in  (read from left to right) that has consecutive repetitions.

Input Format

A single line of input containing the string .

Constraints


Output Format

Print the first occurrence of the repeating character. If there are no repeating characters, print -1.

Sample Input

..12345678910111213141516171820212223
Sample Output

1
Explanation

.. is the first repeating character, but it is not alphanumeric.
1 is the first (from left to right) alphanumeric repeating character of the string in the substring 111.

Python 3



1234567891011
Line: 11 Col: 1
Submit CodeRun Code
Upload Code as File
Test against custom input
Author[deleted]
DifficultyEasy
Max Score20
Submitted By24847
NEED HELP?
View discussions
View editorial
View top submissions
RATE THIS CHALLENGE

MORE DETAILS
Download problem statement
Download sample test cases
Suggest Edits
Share on Facebook
"""
# Enter your code here. Read input from STDIN. Print output to STDOUT

import re
m=re.search(r'([A-Za-z\d])\1{1,}',input())
#m = re.search(r'([A-Za-z0-9])\1+',input())
#print(m)
if m:
    print(m.group(1))
else:
    print(-1)
