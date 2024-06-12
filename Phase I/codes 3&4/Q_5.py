import math
from itertools import permutations


def minimumHamming(z1, z2):
    n = len(z1)
    k = max(z1)

    hamming = []

    Permutations = list(permutations(range(1, k + 1)))

    for i in range(math.factorial(k)):
        output = 0
        for j in range(n):
            index = Permutations[0].index(z1[j])
            if Permutations[i][index] != z2[j]:
                output += 1
        hamming.append(output)

    min_hamming = min(hamming)
    return min_hamming


z1 = [2, 2, 1, 2, 1, 1]
z2 = [1, 1, 2, 1, 2, 2]

# z1 = list(map(int, input().split()))
# z2 = list(map(int, input().split()))

min_hamming = minimumHamming(z1, z2)
print(min_hamming)
