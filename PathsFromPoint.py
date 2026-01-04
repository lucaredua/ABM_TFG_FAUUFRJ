import ghpythonlib.treehelpers as th
import Rhino.Geometry as rg
import random as r

di = {}
for i in range(len(iLines)):
    pt = iLines[i].From
    if pt not in di:
        li = []
        for j in range(len(iLines)):
            if pt == iLines[j].From:
                li.append(iLines[j])
            else: continue
        di[pt] = li
    else: continue

pt = iPoint
listl0 = []
listl1 = []
hist = [len(di[pt])]
histpt = [pt]
while pt in di:
    lin = di[pt][0]
    pt = lin.To
    listl0.append(lin)
    if pt in di:
        hist.append(len(di[pt]))
        histpt.append(pt)
    else:
        listl1.append(listl0)
        listl0 = []
        while len(hist) > 0:
            if hist[-1] == 1:
                hist.pop(-1)
                histpt.pop(-1)
            elif hist[-1] > 1:
                hist[-1] -= 1
                di[histpt[-1]].pop(0)
                pt = iPoint
                hist = [len(di[pt])]
                histpt = [pt]
                break

oLines = th.list_to_tree(listl1)
