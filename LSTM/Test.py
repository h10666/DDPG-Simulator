def maxcount(s1: str, s2: str, n: int):
    dp = [[0]*(n+1) for _ in range(n+1)]
    for i in range(0, n):
        for j in range(0, n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][n]

print(maxcount("ATM", "CAT", 3))