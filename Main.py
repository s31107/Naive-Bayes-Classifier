import sys

from BayesClassifier import BayesClassifier

file_separator = ","

def parse_file(path: str) -> dict[str, list[str]]:
    with open(path, mode="r", encoding="UTF-8") as file:
        attributes = [attr.removesuffix("\n") for attr in file.readline().split(file_separator)]
        attributes_values = {attr: [] for attr in attributes}
        for line in file.readlines():
            for value_index, value in enumerate(line.split(file_separator)):
                attributes_values[attributes[value_index]].append(value.removesuffix("\n"))
        return attributes_values

train_content = parse_file(sys.argv[1])
test_content = parse_file(sys.argv[2])

decision_attribute = tuple(set(train_content) - set(test_content))
assert len(decision_attribute) == 1
bayes_classifier = BayesClassifier(train_content, decision_attribute[0])

for index in range(len(next(iter(test_content.values())))):
    print(bayes_classifier.compute({value : key[index] for value, key in test_content.items()}))

classify_attributes = {}
while True:
    for attribute in test_content:
        classify_attributes.update({attribute : input(f"{attribute}: ")})
    print(bayes_classifier.compute(classify_attributes))
