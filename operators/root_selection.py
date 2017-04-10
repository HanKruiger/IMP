from model import Dataset


def root_selection(source):
    data = Dataset.root.data()[source.indices(), :]
    dataset = Dataset(data, source.indices(), name='Root selection')
    return dataset
