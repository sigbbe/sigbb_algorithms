def dataFromFile(file_name):
    # """Function which reads from the file and yields a generator"""
    file_iter = open(file_name, 'r')
    temp = list()
    for line in file_iter:
        if (len(line.split(',')) < 3):
            continue
        line = line.strip().rstrip(',')  # Remove trailing comma
        temp.append(line)
        # record = frozenset(line.split(','))
    file_iter.close()
    return temp


def createDict(data):
    data = data.split(',')
    return {
        'tid': data[0],
        'items': data[1::]
    }


def a_priori(data):
    pass


def main():
    data = dataFromFile("dataset_oeving2.csv")
    data = list(map(createDict, data))
    print(data)


def countRegex(data, regex):
    data = data.split("\n")[2:]
    for d in data:
        print(d)


inpt = """
tid,items
110,A,B,C,F,G,H
111,A,B,C,E,G
112,C,E,F,H
113,A,B,C,G,H
114,C,D,E,H
115,B,C,E,G
116,A,B,C,D,G,H
117,B,C,E,G
118,A,C,G,H
119,A,B,C,D,E,G,H
"""
if __name__ == '__main__':
    # items = ["A", "B", "C", "D", "E", "F", "G", "H"]
    # rx = re.compile("sdf", re.VERBOSE)
    # items = ["A[a-z,]*C"]
    # for item in items:
    #     print(f"count({item}): {inpt.count(item)}")
    import re

    output = []

    # regex = r'"(.*?)"\s(\d{3})'
    regex = r'"^[0-9]{3}'

    value = re.findall(regex, inpt)
    output.append(value)

    print(output)
