def Testprocess():
    import pandas as  pd
    import numpy as np
    fpath = "./data/test.txt"
    train = pd.read_csv(
        fpath,
        sep=' ',
        header=None,
    )
    # read the positive samples
    test = []
    for i in range(len(train)):
        aa = train.loc[i, 0:75].tolist()
        aa = list(map(float, aa))
        test.append(aa)
    pos=[1 for i in range(54)]
    neg=[0 for i in range(122)]
    #store label
    label =pos+neg
    test = np.array(test)
    label = np.array(label)
    return test,label