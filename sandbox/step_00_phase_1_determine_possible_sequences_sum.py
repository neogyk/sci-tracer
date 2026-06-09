from typing import List

def integer_partitions(n: int) -> List[List[int]]:
    """Generate all partitions of an integer n."""

    def _partitions(num: int, max_value: int) -> List[List[int]]:
        if num == 0:
            return [[]]
        result = []
        for i in range(1, max_value + 1):
            if num >= i:
                for tail in _partitions(num - i, i):
                    result.append([i] + tail)
        return result

    return _partitions(n, n)

def main():
    partitions_of_12 = integer_partitions(12)
    for partition in partitions_of_12:
        print(partition)

if __name__ == "__main__":
    main()