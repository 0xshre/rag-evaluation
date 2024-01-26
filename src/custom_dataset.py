from dspy.datasets.dataset import Dataset
from dspy.primitives.example import Example
import random


class CustomQADataSet(Dataset):
    def __init__(self, queries, answers, train_size=0.7, dev_size=0.15, test_size=0.15):
        super().__init__()
        full_data = self._create_dataset(queries, answers)
        self._train, self._dev, self._test = self._split_dataset(full_data, train_size, dev_size, test_size)

    def _create_dataset(self, queries, answers):
        dataset = []
        for id, question in queries.items():
            answer = answers.get(id, None)
            if answer is not None:
                example = Example(question=question, answer=answer)
                dataset.append(example)
        return dataset

    def _split_dataset(self, dataset, train_size, dev_size, test_size):
        random.shuffle(dataset)
        total = len(dataset)
        train_end = int(total * train_size)
        dev_end = train_end + int(total * dev_size)

        train = dataset[:train_end]
        dev = dataset[train_end:dev_end]
        test = dataset[dev_end:]
        return train, dev, test

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test