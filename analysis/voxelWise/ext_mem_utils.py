import numpy as np
import uuid, os


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

def delete_lines(input_filepath, indeces_to_delete):
    """
    Delete selected lines by index without loading whole file into RAM

    Args:
        input_filepath: path of file to delete lines from
        indeces_to_delete: np.array of indeces to delete

    Returns: undefined
    """
    line_count = 0
    input_dir = os.path.dirname(os.path.abspath(input_filepath))
    temp_file_path = os.path.join(input_dir, str(uuid.uuid4()))
    if os.path.isfile(temp_file_path):
        # file exists
        print('Stopped because overwriting file: ', temp_file_path)

    with open(input_filepath, 'r') as input_file:
        with open(temp_file_path, 'w+') as temp_file:
            for line in input_file:
                if not np.isin(line_count, indeces_to_delete) :
                    temp_file.write(line)
                line_count += 1

    os.remove(input_filepath)
    os.rename(temp_file_path, input_filepath)
