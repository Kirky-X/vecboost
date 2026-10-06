#!/usr/bin/env python3
"""字符错误率 CER：规范化（去空白/标点、拉丁转小写）后按字符 Levenshtein 计算，输出 S/I/D 明细。"""
import sys, re, unicodedata

def normalize(t: str) -> str:
    t = t.lower()
    out = []
    for ch in t:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            continue
        out.append(ch)
    return "".join(out)

def levenshtein(a: str, b: str):
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    dp = [[0]*(m+1) for _ in range(n+1)]
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        dp[i][0] = i
        for j in range(1, m+1):
            c = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
    # 回溯统计 S/I/D
    i, j, S, I, D = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if a[i-1]==b[j-1] else 1):
            S += 0 if a[i-1]==b[j-1] else 1
            i, j = i-1, j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return dp[n][m], S, I, D

ref, hyp = sys.argv[1], sys.argv[2]
r, h = normalize(open(ref, encoding="utf-8").read()), normalize(open(hyp, encoding="utf-8").read())
dist, S, I, D = levenshtein(r, h)
print(f"ref_chars={len(r)} hyp_chars={len(h)}")
print(f"CER={dist/len(r):.4f} (dist={dist})")
print(f"sub={S} ins={I} del={D} (sub_rate={S/len(r):.4f} ins_rate={I/len(r):.4f} del_rate={D/len(r):.4f})")
