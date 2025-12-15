# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 06:50:45 2024

@author: maki
"""


"""
このプログラムは、大きな整数の素数判定を
シングルスレッド / マルチプロセス / マルチスレッド
で実行し、実行時間を比較するテスト。

・Single-thread: 119 seconds
・ProcessPoolExecutor:121 seconds
・ThreadPoolExecutor: 124 seconds

今回は、3つの方式すべてがほぼ同じ実行時間となったが、これは仕様通りの正常な結果である。

理由は以下の通り。
・処理が極端に重いため並列化のメリットが相殺された
・ProcessPool の起動・通信コストが大きい
・ThreadPool は GIL により CPUバウンド処理を並列にできない


-----
------------------------------
Single-thread execution time: 53 seconds
ProcessPoolExecutor execution time: 51 seconds
ThreadPoolExecutor execution time: 52 seconds


"""

import concurrent.futures
import time


# より大きな数値リストを設定
numbers = [
    112272535095293,  # 元の数値
    112582705942171,  # 元の数値
    115280095190773,  # 元の数値
    115797848077099,  # 元の数値
    109972689928541,  # 元の数値
    2305843009213693951  # 非常に大きな数

]

# 素数判定関数
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# シングルスレッドでの実行
def check_primes_single_thread(numbers):
    start_time = time.time()
    results = [is_prime(n) for n in numbers]
    end_time = time.time()
    print("Single-thread execution time:", end_time - start_time, "seconds")
    return results

# ProcessPoolExecutor での実行（マルチプロセス）
def check_primes_process_pool(numbers):
    start_time = time.time()
    # プロセス数をCPUコア数に設定
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(is_prime, numbers))

    end_time = time.time()
    print("ProcessPoolExecutor execution time:", end_time - start_time, "seconds")
    return results


# ThreadPoolExecutor での実行（マルチスレッド）
def check_primes_thread_pool(numbers):
    start_time = time.time()
    #with concurrent.futures.ThreadPoolExecutor() as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(is_prime, numbers))
    end_time = time.time()
    print("ThreadPoolExecutor execution time:", end_time - start_time, "seconds")
    return results

# メインの処理
if __name__ == '__main__':
    print("------------------------------")
    print("シングルスレッド開始:")
    check_primes_single_thread(numbers)

    print("------------------------------")
    print("ProcessPoolExecutor開始:")
    check_primes_process_pool(numbers)

    print("------------------------------")
    print("ThreadPoolExecutor開始:")
    check_primes_thread_pool(numbers)
