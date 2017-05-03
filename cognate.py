import argparse
from xygram import XYGram
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import DictVectorizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lang1', help="Epitran code of src language")
    parser.add_argument('lang2', help="Epitran code of dest language")
    parser.add_argument('file1', help="Path to list of entities in src lang")
    parser.add_argument('file2', help="Path to list of entities in dest lang")
    parser.add_argument('-mo', '--offset', type=int, default=3, help="Maximum length of n-grams")
    parser.add_argument('-mf', '--features', type=int, default=3, help="Maximum number of features grouped")
    parser.add_argument('-d', '--distance', default="cosine", help="Distance metric used for serach")
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

    vec = DictVectorizer()
    V = vec.fit_transform(xygrams1 + xygrams2)
    V1 = V[:len(strings1)]
    V2 = V[len(strings1):]

    try:
        neigh = NearestNeighbors(args.recall, args.distance)
    except:
        print "Invalid distance function. Defaulting to Euclidean distance.\n"
        neigh = NearestNeighbors(args.recall)
    neigh.fit(V2)
    cognates = neigh.kneighbors(V1, return_distance=False)

    # Calculate recall and print out results
    recalled = 0.0
    for i, s in enumerate(strings1):
        print s.encode('utf-8') + " :: " + strings2[i].encode('utf-8')
        for j in cognates[i]:
            print "  - " + strings2[j].encode('utf-8')
        if i in cognates[i]:
            recalled += 1.0

    print "\nRECALL: {0:.1f}%\n".format(recalled / len(strings1) * 100)

main()