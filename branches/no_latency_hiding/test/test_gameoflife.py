import numpy as np
import random
import dnumpytest

def gameoflife(W,H,ITER,DIST,random_state):
    random.setstate(random_state)
    LIVING_LOW= 2
    LIVING_HIGH = 3
    ALIVE = 3

    full      = np.zeros((W+2,H+2), dtype=np.long, dist=DIST)
    dead      = np.zeros((W,H),     dtype=np.long, dist=DIST)
    live      = np.zeros((W,H),     dtype=np.long, dist=DIST)
    live2     = np.zeros((W,H),     dtype=np.long, dist=DIST)
    neighbors = np.zeros((W,H),     dtype=np.long, dist=DIST)

    cells = full[1:W+1,1:H+1]
    ul = full[0:W, 0:H]
    um = full[0:W, 1:H+1]
    ur = full[0:W, 2:H+2]
    ml = full[1:W+1, 0:H]
    mr = full[1:W+1, 2:H+2]
    ll = full[2:W+2, 0:H]
    lm = full[2:W+2, 1:H+1]
    lr = full[2:W+2, 2:H+2]

    for i in xrange(W):
      for j in range(H):
          if random.random() > .8:
              cells[i][j] = 1
    for i in xrange(ITER):
        # zero neighbors
        np.bitwise_and(neighbors,0,neighbors)
        # count neighbors
        neighbors += ul
        neighbors += um
        neighbors += ur
        neighbors += ml
        neighbors += mr
        neighbors += ll
        neighbors += lm
        neighbors += lr
        # extract live cells neighbors
        np.multiply(neighbors, cells, live)
        # find all living cells among the already living
        np.equal(live, LIVING_LOW, live2)
        np.equal(live, LIVING_HIGH, live)
        # merge living cells into 'live'
        np.bitwise_or(live, live2, live)
        # extract dead cell neighbors
        np.equal(cells, 0, dead)
        dead *= neighbors
        np.equal(dead,ALIVE,dead)
        # make sure all threads have read their values
        np.bitwise_or(live, dead, cells)
    return full

def run():
    random_state = random.getstate()
    Seq = gameoflife(100,100,5,False,random_state)
    Par = gameoflife(100,100,5,True,random_state)
    if not dnumpytest.array_equal(Seq,Par):
        raise Exception("Uncorrect result matrix\n")

if __name__ == "__main__":
    run()
