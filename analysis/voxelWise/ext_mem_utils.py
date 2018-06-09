# As inspired from https://stackoverflow.com/a/9312702/3903778

def save_to_svmlight(data, labels, path):
    """
    Save to an svmlight / libsvm file
    This format is a text-based format, with one sample per line.
    The first element of each line can be used to store a target variable to predict.

    Advantage over sklearn.datasets.dump_svmlight_file: it is possible to append to an existing file

    Args:
        path: to save the file to
        data: X
        labels: y

    Returns: undefined
    """

    try:
        file = open(path, 'a')
    except IOError:
        file = open(path, 'w')

    for i, x in enumerate(data):
        indexes = x.nonzero()[0]
        values = x[indexes]

        label = '%i'%(labels[i])
        pairs = ['%i:%f'%(indexes[i] + 1, values[i]) for i in range(len(indexes))]

        sep_line = [label]
        sep_line.extend(pairs)
        sep_line.append('\n')

        line = ' '.join(sep_line)

        file.write(line)
