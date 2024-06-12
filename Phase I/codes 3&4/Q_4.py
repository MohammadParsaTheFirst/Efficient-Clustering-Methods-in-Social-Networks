
z1 = list(map(int, input().split()))
z2 = list(map(int, input().split()))

n = len(z1)
output = 0

for i in range(n):
    if z1[i] != z2[i]:
        output += 1

print("dH = " + str(output))
