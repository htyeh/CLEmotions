from collections import defaultdict

lines = []
with open('en_train.tsv') as test_file:
    for line in test_file:
        lines.append(len(line.split('\t')[0].split()))

len_dict = defaultdict(int)
for item in lines:
    len_dict[item] += 1

len_dict = sorted(len_dict.items(), key=lambda x: x[0])
for k, v in len_dict:
    print(k, v)

print('avg len:', sum(lines)/len(lines))