import numpy as np

def distance(p1,p2):

    dx = max([p1[0],p2[0]]) - min([p1[0], p2[0]])
    dy = max([p1[1],p2[1]]) - min([p1[1], p2[1]])
    d = np.sqrt(dy ** 2 + dx **2)
    return (round(d,2))

def midpoint(p1,p2):
    x = (p1[0] + p2[0])/2
    y = (p1[1] + p2[1])/2
    return(round(x,2),round(y,2))

