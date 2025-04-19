import math

p1 = 0.65
p2 = 0.05
p3 = 0.3

sum = 0
for p in [p1, p2, p3]:
    sum += p*math.log2(p)

print(sum)