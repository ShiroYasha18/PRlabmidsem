def match_pattern(input_pattern, patterns):
    for pattern in patterns:
        if all(x in pattern for x in input_pattern):
            return True
    return False

patterns = [(0,1,1), (1,0,0)]
print(match_pattern((0,1), patterns))  # True for partial match
print(match_pattern((0,1,1), patterns))  # True