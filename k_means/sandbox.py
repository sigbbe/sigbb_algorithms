

def main():
    test = [None, 0, 1, 2, 3, 4, 5]
    for i in range(1, len(test) - 1):
        print(i - 1, test[i + 1:])
    return None


if __name__ == '__main__':
    main()
