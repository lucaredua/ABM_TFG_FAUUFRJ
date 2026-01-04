import Rhino.Geometry as rg
import random as r

class DLAMesh:

    def __init__(self, meshbase, indexbase):

        self.MeshBase = meshbase
        self.IndexBase = indexbase

    def ChosenPoint(self):

        cpoint = self.MeshBase.Faces.GetFaceCenter(self.IndexBase)
        return cpoint

    def Neighbors(self, index):

        fcs = self.MeshBase.Faces.AdjacentFaces(index)

        viz = []
        for i in range(len(fcs)): viz.append(fcs[i]) #array to list

        if len(viz) == 2:
            k = self.MeshBase.Faces.AdjacentFaces(viz[0])
            w = self.MeshBase.Faces.AdjacentFaces(viz[1])
            for i in range(len(k)):
                for j in range(len(w)):
                    if k[i] == w[j] and k[i] != index: viz.append(k[i])
                    else: continue

        elif len(viz) == 3:
            k = self.MeshBase.Faces.AdjacentFaces(viz[0])
            w = self.MeshBase.Faces.AdjacentFaces(viz[1])
            z = self.MeshBase.Faces.AdjacentFaces(viz[2])
            for i in range(len(k)):
                for j in range(len(w)):
                    if k[i] == w[j] and k[i] != index: viz.append(k[i])
                    else: continue
            for i in range(len(k)):
                for j in range(len(z)):
                    if k[i] == z[j] and k[i] != index: viz.append(k[i])
                    else: continue
            for i in range(len(z)):
                for j in range(len(w)):
                    if z[i] == w[j] and z[i] != index: viz.append(z[i])
                    else: continue

        elif len(viz) == 4:
            k = self.MeshBase.Faces.AdjacentFaces(viz[0])
            w = self.MeshBase.Faces.AdjacentFaces(viz[1])
            z = self.MeshBase.Faces.AdjacentFaces(viz[2])
            q = self.MeshBase.Faces.AdjacentFaces(viz[3])
            #K
            for i in range(len(k)):
                for j in range(len(w)):
                    if k[i] == w[j] and k[i] != index: viz.append(k[i])
                    else: continue
            for i in range(len(k)):
                for j in range(len(z)):
                    if k[i] == z[j] and k[i] != index: viz.append(k[i])
                    else: continue
            for i in range(len(k)):
                for j in range(len(q)):
                    if k[i] == q[j] and k[i] != index: viz.append(k[i])
                    else: continue
            #W
            for i in range(len(w)):
                for j in range(len(z)):
                    if w[i] == z[j] and w[i] != index: viz.append(w[i])
                    else: continue
            for i in range(len(w)):
                for j in range(len(q)):
                    if w[i] == q[j] and w[i] != index: viz.append(w[i])
                    else: continue
            #Z
            for i in range(len(z)):
                for j in range(len(q)):
                    if z[i] == q[j] and z[i] != index: viz.append(z[i])
                    else: continue

        return viz

    def EdgeFaces(self):

        edgeFaces = []
        for i in range(len(self.MeshBase.Faces)):
            verif = self.Neighbors(i)
            if len(verif) < 8:
                edgeFaces.append(i)
            else:
                continue
        return edgeFaces

    def Direction(self, index):

        pos = self.MeshBase.Faces.GetFaceCenter(index)
        npos = self.Neighbors(index)
        ori = self.ChosenPoint()
        lin0 = rg.Line(pos, ori)
        vec0 = lin0.Direction
        vec0.Unitize()

        vecs = []
        for i in range(len(npos)):
            cpos = iMesh.Faces.GetFaceCenter(npos[i])
            line = rg.Line(pos, cpos)
            vec = line.Direction
            vec.Unitize()
            vecs.append(vec)

        ang = []
        for i in range(len(vecs)):
            ang.append(rg.Vector3d.VectorAngle(vec0, vecs[i]))

        list00 = []
        for i in range(len(ang)):
            list00.append([ang[i], i])
        list00.sort()

        sindex = []
        for i in list00:
            sindex.append(i[1])

        opt = []
        for i in range(3): opt.append(npos[sindex[i]])

        rnum = r.randint(1,100)
        if 0 < rnum <= 50: return opt[0]
        if 50 < rnum <= 80: return opt[1]
        else: return opt[2]

    def Point(self, index):
        Point = self.MeshBase.Faces.GetFaceCenter(index)
        return Point

# Main Script

oLines = []
pPoint = []

if iReset:
    num = iChosenFace
    mesh = DLAMesh(iMesh, num)
    oPoint = mesh.ChosenPoint()
    vnum = mesh.Neighbors(num)
    
    di = {}
    di[num] = 1
    list0 = [num]
    list1 = [list0]
    ver1 = True
    ver2 = True
    k = 0

else:
    k += 1
    if ver1 == True:
        r.seed(k+seed)
        ind = r.choice(mesh.EdgeFaces())
        hist = [ind]
        ver1 = False
        oPoint = mesh.ChosenPoint()
    else:
        oPoint = mesh.ChosenPoint()
        pPoint.append(mesh.Point(ind))
        upd = mesh.Direction(ind)
        ind = upd
        hist.append(ind)
        print hist
        if ind in di:
            ver1 = True
            pt1 = mesh.Point(ind)
            ind = hist[-2]
            pt2 = mesh.Point(ind)
            di[ind] = 1
            line = rg.Line(pt1, pt2)
            oLines.append(line)
