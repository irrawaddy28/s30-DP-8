'''
413 Arithmetic Slices
https://leetcode.com/problems/arithmetic-slices/description/

An integer array is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.

For example, [1,3,5,7,9], [7,7,7,7], and [3,-1,-5,-9] are arithmetic sequences.

Given an integer array nums, return the number of arithmetic subarrays of nums.

A subarray is a contiguous subsequence of the array.

Example 1:
Input: nums = [1,2,3,4]
Output: 3
Explanation: We have 3 arithmetic slices in nums: [1, 2, 3], [2, 3, 4] and [1,2,3,4] itself.

Example 2:
Input: nums = [1]
Output: 0

Constraints:
1 <= nums.length <= 5000
-1000 <= nums[i] <= 1000

Solution:
1. Brute Force
Count all arithmetic slices starting from index i, where i = 0,...,N-1
eg. nums = [1,2,3,4,5,8]
i=0, slices = [1,2,3], [1,2,3,4], [1,2,3,4,5]
i=1, slices = [2,3,4], [2,3,4,5]
i=2, slices = [3,4,5]
Total = 6 artimetic slices
for i = 0,...,N-3, # O(N)
    for j = i+2,...,N-1, # O(N)
       slice = [i,i+1,...,j]
       if slice is arithmetic
          total += 1
https://youtu.be/sK8MUV3ruzg?t=313
Time: O(N^2), Space: O(1)

2. Tabulation 1 (bottom-up approach)
In brute force, we have repeated subproblems. For eg, in Brtue force, when i = 0, slice = [1,2,3,4,5] (after this, we know that 5-4=1, 4-3=1, 3-2=1)
When i = 1, we get slice = [2,3,4,5] (after recomputing 5-4 = 4-3 = 3-2)
We have alreacy computed this when i = 0. Hence, to avoid solving repeated subproblems, we use DP bottom up.

Since it is bottom up, we go through the array right to left starting from N-2 element. That is, we form a triplet of nums[i], nums[i+1], nums[i+2], where i = N-3. If the difference stays the same, we extend the slice. We add the count of new slices at each step.
Define dp[i] = no. of arithmetic slices possible starting with nums[i]
https://youtu.be/sK8MUV3ruzg?t=935
Time: O(N), Space: O(N)

3. Tabulation 2 (bottom-up approach)
This approach is the easiest to understand.

We read the nums array from left to right, starting at index 2.
For each index i, we compare the first order difference
diff1 = nums[i] - nums[i-1]  (curr - prev)
diff2 = nums[i-1] - nums[i-2] (prev - prev's prev)
If the diffs match, then we have a single arithmetic triplet
(nums[i], nums[i-1], nums[i-2])

dp[i] = no. of arithmetic slices possible ending with nums[i]
      = 1 (triplet) + dp[i-1]

Why add dp[i-1]? Because we could have arithmetic slices ending at nums[i-1].
We take each of those slices (out of a total of dp[i-1]) and append nums[i] to form longer slices.
eg. nums = [1, 2, 3, 4, 5], dp = [0,0,0,0,0]

i = 2, slices  = [1,2,3],
       dp = [0,0,1,0]

i = 3, slices =  [2,3,4], [1,2,3] + [4]
              =  [2,3,4], [1,2,3,4]
       dp = [0,0,1,2]

i = 4, slices = [3,4,5], ([2,3,4], [1,2,3,4]) + [5]
              = [3,4,5], [2,3,4,5], [1,2,3,4,5]
       dp = [0,0,1,2,3]

https://youtu.be/sK8MUV3ruzg?t=1660
Time: O(N), Space: O(N)

4. Tabulation 3 (bottom-up approach)
This is similar to Tabulation 2 except that we use curr and prev variables to instead of a dp array.
https://youtu.be/sK8MUV3ruzg?t=2013
Time: O(N), Space: O(1)
'''
from typing import List

def numberOfArithmeticSlicesBruteForce(nums: List[int]) -> int:
    if not nums or len(nums) < 3:
        return 0
    N = len(nums)
    count = 0
    for i in range(N-2):
        diff = nums[i+1] - nums[i]
        j = i + 1
        while j < N-1: # j < N-1 becausewe use nums[j+1] in the loop
            if nums[j+1] - nums[j] == diff:
                count += 1
            else:
                break
            j += 1
    return count

def numberOfArithmeticSlicesDP1(nums: List[int]) -> int:
    if not nums or len(nums) < 3:
        return 0
    N = len(nums)
    dp = [0]*N
    count = 0
    for i in range(N-3, -1, -1):
        diff1 = nums[i+2] - nums[i+1]
        diff2 = nums[i+1] - nums[i]
        if diff1 == diff2:
            dp[i] = 1 + dp[i+1]
        count += dp[i]
    return count

def numberOfArithmeticSlicesDP2(nums: List[int]) -> int:
    if not nums or len(nums) < 3:
        return 0
    N = len(nums)
    dp = [0]*N
    count = 0
    for i in range(2, N):
        diff1 = nums[i] - nums[i-1]
        diff2 = nums[i-1] - nums[i-2]
        if diff1 == diff2:
            dp[i] = 1 + dp[i-1]
        count += dp[i]
    return count

def numberOfArithmeticSlicesDP3(nums: List[int]) -> int:
    if not nums or len(nums) < 3:
        return 0
    N = len(nums)
    prev, curr = 0, 0
    count = 0
    for i in range(2, N):
        diff1 = nums[i] - nums[i-1]
        diff2 = nums[i-1] - nums[i-2]
        if diff1 == diff2:
            curr = prev + 1 # dp[i] = 1 + dp[i-1]
            count += curr # count += dp[i]
            prev = curr
        else:
            prev = 0
    return count

def run_numberOfArithmeticSlices():
    tests = [([3,5,7,9,11,15,20,25,28,29], 7),
             ([1,2,3,4], 3),
             ([1], 0),
    ]
    for test in tests:
        nums, ans = test[0], test[1]
        print(f"\nnums = {nums}")
        for method in ['brute-force', 'dp-right-to-left', 'dp-left-to-right','dp-left-to-right-memoptim']:
            if method == 'brute-force':
                num_slices = numberOfArithmeticSlicesBruteForce(nums)
            elif method == 'dp-right-to-left':
                num_slices = numberOfArithmeticSlicesDP1(nums)
            elif method == 'dp-left-to-right':
                num_slices = numberOfArithmeticSlicesDP2(nums)
            elif method == 'dp-left-to-right-memoptim':
                num_slices = numberOfArithmeticSlicesDP3(nums)
            print(f"Method {method}: num slices = {num_slices}")
            success = (ans == num_slices)
            print(f"Pass: {success}")
            if not success:
                print(f"Failed")
                return

run_numberOfArithmeticSlices()