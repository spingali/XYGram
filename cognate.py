import sys
from xygram import XYGram
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import DictVectorizer

def main():
    if (len(sys.argv) != 5):
        return None

    xy = XYGram(sys.argv[1], sys.argv[2])

    f = open(sys.argv[3], 'r')
    lines1 = f.readlines()
    f.close()

    f = open(sys.argv[4], 'r')
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

    neigh = NearestNeighbors(10)
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

    print "\nRECALL: {0:.2f}%\n".format(recalled / len(strings1) * 100)

main()