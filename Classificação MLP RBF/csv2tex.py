import numpy

for d in range(1,11):

    data = numpy.genfromtxt('confmatrix-mlp-200-total-%d.csv' % d, delimiter=',')

    # for i, row in enumerate(data):
    #     row = list(map(int,row))
    #     print('&%02d \\vrule & %s \\\\' % (i+1, ' & '.join(list(map(str,row)))))

    recall = [data[i][i]/sum([data[i][j] for j in range(30)]) for i in range(30)]
    precision = [data[i][i]/sum([data[j][i] for j in range(30)]) for i in range(30)]

    print(numpy.mean(recall))
    print(numpy.mean(precision))
    print()