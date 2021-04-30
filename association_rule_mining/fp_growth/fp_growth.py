import pandas as pd
from pyspark.ml.fpm import FPGrowth
from pyspark.python.pyspark.shell import spark


def count_of_element(ls, el):
    count = 0
    for row in ls:
        print(row)
        if (el in row[1]):
            count += 1
    return count


if __name__ == "__main__":
    itemSetList = []
    data = pd.read_csv('./data/oeving_5_with_fields.csv',
                       delimiter=',', keep_default_na=False)
    print(itemSetList)
    # df = spark.createDataFrame([
    #     (110, ["A", "B", "C", "F", "G", "H"]),
    #     (111, ["A", "B", "C", "E", "G"]),
    #     (112, ["C", "E", "F", "H"]),
    #     (113, ["A", "B", "C", "G", "H"]),
    #     (114, ["C", "D", "E", "H"]),
    #     (115, ["B", "C", "E", "G"]),
    #     (116, ["A", "B", "C", "D", "G", "H"]),
    #     (117, ["B", "C", "E", "G"]),
    #     (118, ["A", "C", "G", "H"]),
    #     (119, ["A", "B", "C", "D", "E", "G", "H"])
    # ], ["id", "items"])

    print('Count of "P":', count_of_element(itemSetList, 'P'))

    df = spark.createDataFrame([(x[0], x[1:])
                               for x in itemSetList], ["id", "items"])

    fpGrowth = FPGrowth(itemsCol="items", minSupport=.6, minConfidence=.0)
    model = fpGrowth.fit(df)

    # Display frequent itemsets.
    model.freqItemsets.show()

    # Display generated association rules.
    # model.associationRules.show()

    # transform examines the input items against all the association rules and summarize the
    # consequents as prediction
    model.transform(df).show()
