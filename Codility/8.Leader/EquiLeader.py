"""
A non-empty array A consisting of N integers is given.

The leader of this array is the value that occurs in more than half of the elements of A.

An equi leader is an index S such that 0 ≤ S < N − 1 and two sequences A[0], A[1], ..., A[S] and A[S + 1], A[S + 2], ..., A[N − 1] have leaders of the same value.

For example, given array A such that:

    A[0] = 4
    A[1] = 3
    A[2] = 4
    A[3] = 4
    A[4] = 4
    A[5] = 2
we can find two equi leaders:

0, because sequences: (4) and (3, 4, 4, 4, 2) have the same leader, whose value is 4.
2, because sequences: (4, 3, 4) and (4, 4, 2) have the same leader, whose value is 4.
The goal is to count the number of equi leaders.

Write a function:

def solution(A)

that, given a non-empty array A consisting of N integers, returns the number of equi leaders.

For example, given:

    A[0] = 4
    A[1] = 3
    A[2] = 4
    A[3] = 4
    A[4] = 4
    A[5] = 2
the function should return 2, as explained above.
"""


# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(A):
    # write your code in Python 3.6
    if len(A) == 1:
        return 0
    value = A[0]
    size = 0
    for i in range(len(A)):
        if size == 0:
            size += 1
            value = A[i]
        else:
            if A[i] == value:
                size += 1
            else:
                size -= 1
    print(size)
    candidate = -1
    count = 0
    if size > 0:
        candidate = value

    for i in range(len(A)):
        if A[i] == candidate:
            count += 1
    if count <= len(A) // 2:
        return 0
    leader = candidate
    equiCount = 0
    leaderCount = 0

    for i in range(len(A)):
        if A[i] == leader:
            leaderCount += 1
        if leaderCount > (i + 1) // 2 and (count - leaderCount) > (len(A) - i - 1) // 2:
            equiCount += 1

    return equiCount

## https://app.codility.com/demo/results/training6AR55P-PX5/