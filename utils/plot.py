from matplotlib import pyplot as plt


def plot(data):
    shape = data.shape
    figure, axis = plt.subplots(data)
    for d in data:
        axis.scatter()

    return None
