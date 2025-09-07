'''
120 Triangle
https://leetcode.com/problems/triangle/description/

Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.



Example 1:
Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
Explanation: The triangle looks like:
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).

Example 2:
Input: triangle = [[-10]]
Output: -10

Constraints:
1 <= triangle.length <= 200
triangle[0].length == 1
triangle[i].length == triangle[i - 1].length + 1
-10^4 <= triangle[i][j] <= 10^4

Follow up: Could you do this using only O(n) extra space, where n is the total number of rows in the triangle?

Solution:
1. Recursion
We're starting from the top of the triangle and using recursion to go down.
At each step, we take the min of the two possible paths and store it in memo.
This way, we avoid recalculating and get the min path sum from top to bottom.
Time: O(2^N), Space: O(N)

2. Tabulation (bottom-up approach)
We use a 2D dp array to build up the path sums row by row from top to bottom.
For each element, we take the minimum from the two possible paths above.
In the end, we just return the minimum value from the last row of the dp.
Time: O(N^2), Space: O(N^2)

3. Tabulation (bottom-up approach)
We use a 2D dp array to build up the path sums row by row from bottom to top.
At each step, we take the min of the two paths below and add it to the current value. Finally, the top cell holds the minimum path sum.
Time: O(N^2), Space: O(N^2)

We use 1-D DP tabulation to track the min path sum from bottom to top

'''
from typing import List

def minimumTotal(triangle: List[List[int]]) -> int:
    ''' Time: O(2^N), Space: O(N) '''
    def recurse(r, c):
        if r == len(triangle):
            return 0
        if memo[r][c] is not None:
            return memo[r][c]
        case0 = recurse(r + 1, c) + triangle[r][c]
        case1 = recurse(r + 1, c + 1) + triangle[r][c]
        memo[r][c] = min(case0, case1)
        return memo[r][c]

    n = len(triangle)
    memo = [[None] * n for _ in range(n)]
    return recurse(0, 0)

def minimumTotal_dp1(triangle: List[List[int]]) -> int:
    '''
        Top to Bottom
        Time: O(N^2), Space: O(N^2)
    '''
    n = len(triangle)
    dp = [[0] * n for _ in range(n)]
    dp[0][0] = triangle[0][0]

    for i in range(1, n):
        for j in range(i + 1):
            if j == 0:
                dp[i][j] = triangle[i][j] + dp[i-1][0]
            elif j == i:
                dp[i][j] = triangle[i][j] + dp[i-1][j-1]
            else:
                dp[i][j] = triangle[i][j] + min(dp[i-1][j], dp[i-1][j-1])

    return min(dp[n-1])

def minimumTotal_dp2(triangle: List[List[int]]) -> int:
    '''
        Bottom to Top
        Time: O(N^2), Space: O(N^2)
    '''
    if not triangle:
        return 0

    N = len(triangle)
    dp = [ [0]*N for _ in range(N) ]
    for j in range(N):
        dp[N-1][j] = triangle[N-1][j]

    for i in range(N-2,-1,-1):
        for j in range(i+1):
            dp[i][j] = triangle[i][j] + min(dp[i+1][j], dp[i+1][j+1])

    return dp[0][0]

def minimumTotal_dp3(triangle: List[List[int]]) -> int:
    '''
        Bottom to Top with a 1D DP array
        Time: O(N^2), Space: O(N)
    '''
    if not triangle:
        return 0

    N = len(triangle)
    dp = [0]*N
    for j in range(N):
        dp[j] = triangle[N-1][j]

    for i in range(N-2,-1,-1):
        for j in range(i+1):
            dp[j] = triangle[i][j] + min(dp[j], dp[j+1])

    return dp[0]

def minimumTotal_dp4(triangle: List[List[int]]) -> int:
    '''
        Bottom to Top with in-place mutation (reuse triangle as dp matrix)
        Time: O(N^2), Space: O(1)
    '''
    n = len(triangle)
    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1])
    return triangle[0][0]

def run_minimumTotal():
    tests = [([[2],[3,4],[6,5,7],[4,1,8,3]], 11),
             ([[-10]], -10),
             ([[-1],[2,3],[1,-1,-3]], -1),
    ]
    for test in tests:
        triangle, ans = test[0], test[1]
        sum = minimumTotal_dp2(triangle)
        print(f"\ntriangle = {triangle}")
        for method in ['recursion', 'dp1', 'dp2', 'dp3', 'dp4']:
            if method == 'recursion':
                sum = minimumTotal(triangle)
            elif method == 'dp1':
                sum = minimumTotal_dp1(triangle)
            elif method == 'dp2':
                sum = minimumTotal_dp2(triangle)
            elif method == 'dp3':
                sum = minimumTotal_dp3(triangle)
            elif method == 'dp4':
                sum = minimumTotal_dp4(triangle)
            print(f"Method {method}: Min path sum = {sum}")
            success = (ans == sum)
            print(f"Pass: {success}")
            if not success:
                print("Failed")
                return

run_minimumTotal()
