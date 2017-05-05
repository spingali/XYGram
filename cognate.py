import argparse
from xygram import XYGram
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
from sklearn.feature_extraction import DictVectorizer

def cosine_distance(X, Y):
    return 1 - pairwise.cosine_similarity([X], [Y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lang1', help="Epitran code of src language")
    parser.add_argument('lang2', help="Epitran code of dest language")
    parser.add_argument('file1', help="Path to list of entities in src lang")
    parser.add_argument('file2', help="Path to list of entities in dest lang")
    parser.add_argument('-mo', '--offset', type=int, default=3, help="Maximum length of n-grams")
    parser.add_argument('-mf', '--features', type=int, default=3, help="Maximum number of features grouped")
    # parser.add_argument('-d', '--distance', default="cosine", help="Distance metric used for serach")
    parser.add_argument('-r', '--recall', type=int, default=5, help="Number of candidates recalled for cognacy")
    args = parser.parse_args()

    xy = XYGram(args.lang1, args.lang2, args.offset, args.features)

    f = open(args.file1, 'r')
    lines1 = f.readlines()
    f.close()

    f = open(args.file2, 'r')
    lines2 = f.readlines()
    f.close()

    strings1 = [ l.strip().decode('utf-8') for l in lines1 ]
    strings2 = [ l.strip().decode('utf-8') for l in lines2 ]

    xygrams1 = [ xy.generateXYGram(s, 1) for s in strings1 ]
    xygrams2 = [ xy.generateXYGram(s, 2) for s in strings2 ]

    vec = DictVectorizer(sparse=False)
    V = vec.fit_transform(xygrams1 + xygrams2)
    V1 = V[:len(strings1)]
    V2 = V[len(strings1):]
    print str(V1)

    # try:
    neigh = NearestNeighbors(args.recall, algorithm='ball_tree', metric=cosine_distance)
    # except:
    #     print "\nSPACE USAGE: {}\n".format(len(V) * len(V[0]))
    #     print "Invalid distance function. Exiting.\n"
    #     return None
    print "Initialized NearestNeighbors object."
    neigh.fit(V2)
    print "Fit NearestNeighbors."
    cognates = neigh.kneighbors(V1, return_distance=False)
    print "Found neighbors."

    # Calculate recall and print out results
    recalled = 0.0
    for i, s in enumerate(strings1):
        print s.encode('utf-8') + " :: " + strings2[i].encode('utf-8')
        for j in cognates[i]:
            print "  - " + strings2[j].encode('utf-8')
        if i in cognates[i]:
            recalled += 1.0

    print "\nSPACE USAGE: {}\n".format(len(V) * len(V[0]))
    print "\nRECALL: {0:.2f}%\n".format(recalled / len(strings1) * 100)

main()