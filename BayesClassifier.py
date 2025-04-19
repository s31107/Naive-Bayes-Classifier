import itertools


class BayesClassifier:
    def __init__(self, train_data: dict[str, list[str]], decision_attribute: str):
        assert len(set([len(item) for item in train_data.values()])) == 1
        self.train_data = train_data
        self.decision_attribute = decision_attribute

    def laplace_smoothing(self, counter: int, divider: int, attribute_name: str) -> float:
        if counter != 0:
            return counter / divider
        return 1 / (divider + len(set(self.train_data[attribute_name])))

    # P(attribute_name=item_name|self.decision_attribute=decision_item_name)
    def condition_probability(
            self, attribute_name: str, item_name: str, decision_item_name: str, divider: int) -> float:
        return self.laplace_smoothing(len([1 for index, attr in enumerate(self.train_data[attribute_name])
                                           if attr == item_name and self.train_data.get(
                self.decision_attribute)[index] == decision_item_name]), divider, attribute_name)

    def compute(self, classify_data: dict[str, str]) -> str:
        computed_probabilities = dict(zip(self.train_data[self.decision_attribute], itertools.repeat(1)))
        for decision_item in set(self.train_data[self.decision_attribute]):
            decision_attribute_number = self.train_data[self.decision_attribute].count(decision_item)
            computed_probabilities[decision_item] *= decision_attribute_number / len(
                self.train_data[self.decision_attribute])
            for attribute_name, item_name in classify_data.items():
                computed_probabilities[decision_item] *= self.condition_probability(
                    attribute_name, item_name, decision_item, decision_attribute_number)
        return max(computed_probabilities, key=computed_probabilities.get)