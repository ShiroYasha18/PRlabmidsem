#2. Write a program to Match the given input patterns.
import seaborn as sns
def match_pattern(input_pattern, dataset, columns):
    matches = []
    for i, row in dataset[columns].iterrows():
        if list(row.values) == input_pattern:
            matches.append(i)
    return matches if matches else "No exact match found"
titanic = sns.load_dataset('titanic')
print(titanic)
selected_columns = ['sex', 'pclass', 'embarked']
input_pattern = ['female', 1, 'S']
result = match_pattern(input_pattern, titanic, selected_columns)
print(f"Matching row indices: {result}")
