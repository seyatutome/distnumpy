import numpy as np
import sys
import time

DIST=int(sys.argv[1])

W = int(sys.argv[2])
H = int(sys.argv[2])

DISPLAY = False

full = np.zeros((W+2,H+2), dtype=np.double, dist=DIST)
work = np.zeros((W,H), dtype=np.double, dist=DIST)
diff = np.zeros((W,H), dtype=np.double, dist=DIST)

cells = full[1:-1, 1:-1]
up    = full[1:-1, 0:-2]
left  = full[0:-2, 1:-1]
right = full[2:  , 1:-1]
down  = full[1:-1, 2:  ]

full[:,0]  += -273.15
full[:,-1] += -273.15
full[0,:]  +=  40.0
full[-1,:] += -273.13

t1 = time.time()

epsilon=W*H*0.010
delta=epsilon+1
i=0
while epsilon<delta:
  i+=1
  work[:] = cells
  work += up
  work += left
  work += right
  work += down
  work *= 0.2
  np.subtract(cells,work,diff)
  diff=np.absolute(diff)
  delta=np.sum(diff)
  cells[:] = work
  if DISPLAY and i%100==0:
    np.save("%s.%08d"%(sys.argv[3],i), full)
    print epsilon,'<',delta

t2 = time.time()
print "Itterations", i
print "Time spent:", t2-t1
