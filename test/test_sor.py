import numpy as np
import dnumpytest

def SOR(H,W,Dist):
    if W%2 > 0 or H%2 > 0:
        raise Exception("Each dimension must have an even size.")

    full = np.zeros((H+2,W+2), dtype=np.double, dist=Dist)
    diff = np.zeros((H/2,W/2), dtype=np.double, dist=Dist)

    cells  = full[1:-1, 1:-1]
    black1 = full[1:-1:2, 1:-1:2]
    black2 = full[2:-1:2, 2:-1:2]
    red1   = full[1:-1:2, 2:-1:2]
    red2   = full[2:-1:2, 1:-1:2]

    black1_up    = full[0:-3:2, 1:-1:2]
    black2_up    = red1
    black1_right = red1
    black2_right = full[2:-1:2, 3::2]
    black1_left  = full[1:-2:2, 0:-2:2]
    black2_left  = red2
    black1_down  = red2
    black2_down  = full[3::2, 2:-1:2]
    red1_up      = full[0:-2:2, 2:-1:2]
    red2_up      = black1
    red1_right   = full[1:-2:2, 3::2]
    red2_right   = black2
    red1_left    = black1
    red2_left    = full[2:-1:2, 0:-3:2]
    red1_down    = black2
    red2_down    = full[3::2, 1:-2:2]

    full[:,0]  += -273.15
    full[:,-1] += -273.15
    full[0,:]  +=  40.0
    full[-1,:] += -273.13

    epsilon=W*H*0.002
    delta=epsilon+1
    i=0
    while epsilon<delta:
      i+=1
      diff[:] = black1
      black1 += black1_up
      black1 += black1_right
      black1 += black1_left
      black1 += black1_down
      black1 *= 0.2
      diff -= black1
      np.absolute(diff, diff)
      delta = np.sum(diff)

      diff[:] = black2
      black2 += black2_up
      black2 += black2_right
      black2 += black2_left
      black2 += black2_down
      black2 *= 0.2
      diff -= black2
      np.absolute(diff, diff)
      delta += np.sum(diff)

      diff[:] = red1
      red1 += red1_up
      red1 += red1_right
      red1 += red1_left
      red1 += red1_down
      red1 *= 0.2
      diff -= red1
      np.absolute(diff, diff)
      delta += np.sum(diff)

      diff[:] = red2
      red2 += red2_up
      red2 += red2_right
      red2 += red2_left
      red2 += red2_down
      red2 *= 0.2
      diff -= red2
      np.absolute(diff, diff)
      delta += np.sum(diff)
    return cells

def run():
    Seq = SOR(10,10,False)
    Par = SOR(10,10,True)

    if not dnumpytest.array_equal(Seq,Par):
        return (True, "Uncorrect result matrix\n")
    return (False, "")

if __name__ == "__main__":
    print run()
