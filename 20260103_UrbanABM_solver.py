import Rhino.Geometry as rg
import rhinoscriptsyntax as rs
import ghpythonlib.components as gh
import math
import copy
import random as r
import numpy as np
from dataclasses import dataclass
from scipy.spatial import Voronoi
from math import fsum

R = rg.RTree()
pln = rg.Plane.WorldXY


"""
--------------------------------------------------------------------------- Funções
"""

def PointRandomOnLine(line): #1
    """Gera ponto aleatório em uma linha"""
    num = r.uniform(0, 1)
    return line.PointAt(num)

def EntranceExit(listLines): #2
    """Escolha uma linha de entrada e uma linha de saída, retorna pontos de entrada e saída"""
    entranceLine = r.choice(range(len(listLines)))
    entrancePoint = PointRandomOnLine(listLines[entranceLine])
    exitLine = copy.deepcopy(entranceLine)
    while exitLine == entranceLine:
        exitLine = r.choice(range(len(listLines)))
    exitPoint = PointRandomOnLine(listLines[exitLine])
    return entrancePoint, exitPoint

def VectorConstrain(vec, mx=10, mn=0): #3
    """Limita o tamanho do vetor"""
    if vec.Length <= mx and vec.Length >= mn:
        return vec
    elif vec.Length > mx:
        vlen = mx
        vec.Unitize()
        vec *= vlen
        return vec
    else:
        vlen = mn
        vec.Unitize()
        vec *= vlen
        return vec

def VectorRandom(length=1): #4
    """Gera vetor aleatório"""
    x = r.uniform(-1, 1)
    y = r.uniform(-1, 1)
    z = 0
    vec = rg.Vector3d(x, y, z)
    vec.Unitize()
    vec *= length
    return vec

def Vector2Pt(pt0, pt1): #5
    ln = rg.Line(pt1, pt0)
    return ln.Direction

def ForceGrav2Pt(pt0, pt1, selfMass, targetMass, G = 1): #6
    dist = pt0.DistanceTo(pt1)
    repelVec = Vector2Pt(pt1, pt0)
    repelVec.Unitize
    mag = (G * selfMass * targetMass) / (dist * dist)
    repelVec *= mag
    return repelVec

def VoronoiScipyRhino(points): #7
    pts = []
    for p in points:
        if isinstance(p, rg.Point3d):
            pts.append((p.X, p.Y))
        elif rs.IsPoint(p):
            pts.append(rs.PointCoordinates(p)[:2])
        else:
            pts.append(p[:2])
    pts = np.array(pts)

    vor = Voronoi(pts)
    curves, centroids = [], []
    for region in vor.regions:
        if -1 in region or len(region) == 0:
            continue
        pts3d = [rg.Point3d(vor.vertices[i, 0], vor.vertices[i, 1], 0) for i in region] + [rg.Point3d(vor.vertices[region[0], 0], vor.vertices[region[0], 1], 0)]
        pl = rg.Polyline(pts3d)
        curves.append(pl)
    return curves

def bezier_span(t, P0, T0, P3, T3): #8
    def _cubic(_t, a, b, c, d):
        mt = 1.0 - _t
        mt2 = mt * mt
        t2 = _t * _t
        return fsum([            mt2 * mt  * a,
                          3.0 * mt2 * _t  * b,
                          3.0 * mt  * t2  * c,
                                _t  * t2  * d])
    if isinstance(P0, (int, float)):
        return _cubic(t, P0, T0, T3, P3)
    return type(P0)(_cubic(t, a, b, c, d)
                    for a, b, c, d in zip(P0, T0, T3, P3))

def MeshFlowField(mesh, time, edgeSize = 5): #9
    if time == 0:
        trimesh = gh.TriRemesh(mesh, None, False, None, edgeSize, 25)[0]
        outmesh = trimesh
        fcs = trimesh.Faces
        fcnrmls = trimesh.FaceNormals
    else:
        outmesh = mesh
        fcs = mesh.Faces
        fcnrmls = mesh.FaceNormals
    pts = []
    vecs = {}
    coord = {}
    dictvazios = {}
    vecz = rg.Vector3d(0,0,1)
    for i in range(len(fcs)):
        pt = fcs.GetFaceCenter(i)
        pts.append(pt)
        coord[(round(pt.X, 2), round(pt.Y, 2))] = 0
        dictvazios[(round(pt.X, 2), round(pt.Y, 2))] = i
    for i, j in enumerate(fcnrmls):
        nrml3d = rg.Vector3d(j.X, j.Y, j.Z)
        crss = rg.Vector3d.CrossProduct(vecz, nrml3d)
        crss.Z = 0
        crss.Unitize()
        vecs[i] = crss
    return pts, vecs, fcs, coord, dictvazios, outmesh

def getEntries(mesh, listClosedCrvs): #10
    pln = rg.Plane(rg.Point3d(0,0,0), rg.Point3d(1,0,0), rg.Point3d(0,1,0))
    nkdEdgeMesh = mesh.GetNakedEdges()
    transf = rg.Transform.PlanarProjection(pln)
    nkdEdgeMesh[0].Transform(transf)
    projection = nkdEdgeMesh[0]
    lines = projection.GetSegments()
    list02 = []
    for crv in listClosedCrvs:
        list01 = []
        crv.Transform(transf)
        crv = crv.ToPolylineCurve()
        for ind, ln in enumerate(lines):
            pt = ln.PointAt(.5)
            test = crv.Contains(pt)
            if crv.IsClosed and test == rg.PointContainment.Inside:
                list01.append(ind)
        list02.append(list01)
    listEntries = []
    for i in list02:
        crvsToJoin = []
        for j in i:
            crvsToJoin.append(lines[j])
        plln = rg.Polyline.CreateByJoiningLines(crvsToJoin, .001, False)[0]
        plln = plln.ToPolylineCurve()
        entrie = rg.Line(plln.PointAtEnd, plln.PointAtStart)
        listEntries.append(entrie)
    return listEntries, projection

def Delaunay(listPoint, AreaMin, LenMax): #11
    listPlln = []
    delaunay = gh.DelaunayMesh(listPoint, None)
    if delaunay != None:
        faceB = gh.FaceBoundaries(delaunay)
        for i in faceB:
            if i.IsPlanar():
                area = gh.Area(i)[0]
            else: area = AreaMin
            if area < AreaMin:
                plln = i.ToPolyline()
                distList = []
                for j in range(len(plln)-1):
                    dist = plln[j].DistanceTo(plln[j+1])
                    distList.append(dist)
                verif = True
                for j in distList:
                    if j > LenMax: verif = False
                if verif == True: listPlln.append(plln)
    return listPlln

def MergePolylines(listPolylines, dist): #12
    dict0 = {}
    for i, k in enumerate(listPolylines):
        for j in range(len(k)-1):
            tpl_0 = (k[j].X, k[j].Y, k[j].Z)
            tpl_1 = (k[j+1].X, k[j+1].Y, k[j+1].Z)
            chv_0 = (tpl_0, tpl_1)
            chv_1 = (tpl_1, tpl_0)
            if chv_0 not in dict0 and chv_1 not in dict0: dict0[chv_0] = 0
            elif chv_0 in dict0 and chv_1 not in dict0: dict0[chv_0] += 1
            elif chv_0 not in dict0 and chv_1 in dict0: dict0[chv_1] += 1
            else: pass

    listln = []
    for i in dict0:
        if dict0[i] == 0:
            pt0 = rg.Point3d(i[0][0], i[0][1], i[0][2])
            pt1 = rg.Point3d(i[1][0], i[1][1], i[1][2])
            ln = rg.Line(pt0, pt1)
            listln.append(ln)

    listplln = rg.Polyline.CreateByJoiningLines(listln, .01, True)
    listpllnproj = []
    for i in listplln:
        polyline_proj = rg.Polyline()
        for pt in i:              # Projeta a polyline no plano XY
            projected_point = pln.ClosestPoint(pt)
            polyline_proj.Add(projected_point)
        listpllnproj.append(polyline_proj)

    listOutput = []
    for i in listpllnproj:
        pllncrv = i.ToPolylineCurve()
        if pllncrv.ClosedCurveOrientation() == rg.CurveOrientation.CounterClockwise:
            offset = pllncrv.Offset(pln, -dist, .01, rg.CurveOffsetCornerStyle.Sharp)
            if offset != None:
                for j in offset:
                    listOutput.append(j)
        else:
            offset = pllncrv.Offset(pln, dist, .01, rg.CurveOffsetCornerStyle.Sharp)
            if offset != None:
                for j in offset:
                    listOutput.append(j)
    return listOutput

def EmptySpacesPoints(listPoint, listIndex, AreaMin, LenMax, Dist): #13
    listPtEmpty = []
    listIndexEmpty = list(range(len(listPoint)))
    for i in listIndex:
        if i in listIndexEmpty: listIndexEmpty.remove(i)
        else: pass
    for i in listIndexEmpty:
        listPtEmpty.append(listPoint[i])
    if len(listPtEmpty) > 3:
        delaunay = Delaunay(listPtEmpty, AreaMin, LenMax)
        return MergePolylines(delaunay, Dist)
    else: pass

def Skeletonization2D(crv): #14
    pts = []
    lns = []
    plln = crv.ToPolyline()
    for i in plln: pts.append(i)
    pts = rg.Point3d.CullDuplicates(pts, .01)
    voronoi = VoronoiScipyRhino(pts)
    for i in voronoi:
        seg = i.GetSegments()
        for j in seg:
            test1 = crv.Contains(j.From, pln, .01)
            test2 = crv.Contains(j.To, pln, .01)
            if test1 == test2 == rg.PointContainment.Inside:
                lns.append(j)
    return lns

def cullDplSkeletonization2D(crv): #15
    lns = Skeletonization2D(crv)
    rmvDpl = gh.removeDuplicateLines(lns, .01)
    listlns = []
    if rmvDpl == None: pass
    elif type(rmvDpl) == rg.Line:
        return [rmvDpl]
    else:
        for i in rmvDpl: listlns.append(i)
        return listlns

def VerifSkeletonization2D(crv): #16
    if type(crv) == rg.PolylineCurve:
        plln = crv.ToPolyline()
    else:
        crv = None
    if crv != None:
        if plln.IsClosed and plln.SegmentCount > 4: return cullDplSkeletonization2D(crv)
        else: pass
    else: pass

def ListSkeletonization2D(listCrvs): #17
    listOutput = []
    for k in listCrvs:
        listlns = (VerifSkeletonization2D(k))
        if listlns != None: listOutput.append(listlns)
        else: pass
    return listOutput

def PointsFromLines(listLines): #18
    listOutput = []
    for i in listLines:
        if type(i) == list:
            for j in i:
                listOutput.append(j.From)
                listOutput.append(j.To)
        else:
            listOutput.append(j.From)
            listOutput.append(j.To)
    listOutput = rg.Point3d.CullDuplicates(listOutput, .01)
    return listOutput

def AttractXVertices(mesh, path): #19
    lns = ListSkeletonization2D(path)
    pts = PointsFromLines(lns)
    vertices = mesh.Vertices
    ptCloud = rg.PointCloud()
    for i in vertices:
        ptCloud.Add(i)
    if pts != None:
        ptsVert = []
        for i in pts:
            ind = ptCloud.ClosestPoint(i)
            ptsVert.append(vertices[ind])
        listOutput = rg.Point3d.CullDuplicates(ptsVert, .01)
        return listOutput, vertices
    else: return None, None

def MultiPointsMins(mesh, path, valMin=1, height=1): #20
    ptsToAttract, vertices = AttractXVertices(mesh, path)

    if vertices != None:
        listMins = [valMin] * len(vertices)
        
        for j in ptsToAttract:
            for k, w in enumerate(vertices):
                dist = j.DistanceTo(w)
                if dist < listMins[k]:
                    listMins[k] = dist
        bnds = gh.Bounds(listMins)
        dmn = gh.ConstructDomain(1, 0)
        reMapped = gh.RemapNumbers(listMins, bnds, dmn)[0]
        belzierNums = []
        for i in reMapped: belzierNums.append(bezier_span(i, P0, T0, P1, T1)[1] * height)
        return belzierNums
    else:
        return None

def MeshNewTopology(mesh, originalMesh, path, height): #21
    zValues = MultiPointsMins(mesh, path, distMinTopo, height)
    if zValues != None:
        originalMeshVertices = originalMesh.Vertices
        listPtsMoved = []
        nkdLns = mesh.GetNakedEdges()
        for i, j in enumerate(mesh.Vertices):
            pt = rg.Point3d(j.X, j.Y, (originalMeshVertices[i].Z + zValues[i]))
            listPtsMoved.append(pt)
        return rg.Mesh.CreateFromTessellation(listPtsMoved, nkdLns, pln, False)
    else: return mesh

"""
--------------------------------------------------------------------------- Classe Mover
"""

@dataclass
class Mover:
    point: rg.Point3d()
    destiny: rg.Point3d()
    vel: rg.Vector3d()
    acc: rg.Vector3d()
    mass: float
    edges: rg.Polyline()

    def update(self) -> None: #1
        """Atualização da posição da partícula"""
        self.vel = rg.Vector3d.Add(self.vel, self.acc)
        self.vel = VectorConstrain(self.vel, 1, 0)    # <----------
        self.point += self.vel
        self.acc = rg.Vector3d(0, 0, 0)
        self.checkedges()

    def checkedges(self) -> None: #2
        newplln_ = self.edges.ToPolylineCurve()
        test = newplln_.Contains(self.point)
        if newplln_.IsClosed and test == rg.PointContainment.Outside:
            pt = self.edges.ClosestPoint(self.point)
            self.point = pt
        else: pass
    
    def applyForce(self, force) -> None: #3
        """Aplicação da força"""
        acc2 = force * (1/self.mass)
        self.acc = rg.Vector3d.Add(acc2, self.acc)

    def goToDestiny(self, vecMax, vecMin) -> None: #4
        """Comportamento de ir até o destino escolhido"""
        force = ForceGrav2Pt(self.point, self.destiny, self.mass, 1)
        force = VectorConstrain(force, vecMax, vecMin)
        self.applyForce(force)
    
    def attractToPath(self, listPoint, vecMax, vecMin, tol) -> None: #5
        """Comportamento de se atrair pelo caminho criado"""
        if len(listPoint) > 1:
            indices = list(R.Point3dKNeighbors(listPoint, [self.point], 2))
            ind = indices[0][1]
            pt = listPoint[ind]
            dist = self.point.DistanceTo(pt)
            if dist < tol:
                force = ForceGrav2Pt(self.point, pt, self.mass, .01)
                mult = rg.Vector3d.Multiply(self.vel, force)
                if mult >= 0:
                    force = VectorConstrain(force, vecMax, vecMin)
                    self.applyForce(force)
                else:
                    pass
        else: pass
    
    def repelPath(self, listPoint, vecMax, vecMin, tol) -> None: #6
        """Comportamento de se repelir do caminho criado"""
        if len(listPoint) > 1:
            indices = list(R.Point3dKNeighbors(listPoint, [self.point], 2))
            ind = indices[0][1]
            pt = listPoint[ind]
            dist = self.point.DistanceTo(pt)
            if dist < tol:
                force = ForceGrav2Pt(self.point, pt, self.mass, .01)
                mult = rg.Vector3d.Multiply(self.vel, force)
                if mult >= 0:
                    force = VectorConstrain(force, vecMax, vecMin)
                    self.applyForce(-force)
                else:
                    pass
        else: pass
    
    def flowFieldFollowing(self, listPoint, listVector, vecMax, vecMin): #7
        """Comportamento de ir ser influenciado pela topografia"""
        indices = list(R.Point3dKNeighbors(listPoint, [self.point], 2))
        ind = indices[0][1]
        vec = listVector[ind]
        force = rg.Vector3d.Subtract(self.vel, vec)
        force = VectorConstrain(force, vecMax, vecMin)
        self.applyForce(force)

    def repelClosest(self, moverList, vecMax, vecMin) -> None: #8
        """Comportamento de afastamento do mais próximo"""
        indices = list(R.Point3dKNeighbors(moverList, [self.point], 2))
        ind = indices[0][1]
        dist = self.point.DistanceTo(moverList[ind])
        if dist < .25:
            force = ForceGrav2Pt(moverList[ind], self.point, self.mass, 1)
            force = VectorConstrain(force, vecMax, vecMin)
            self.applyForce(force)
    
    def repelFromObstacle(self, obsts, scalefactor) -> None: #9
        """Comportamento de afastamento dos obstáculos (closed curves)"""
        for i in obsts:
            center = i.CenterPoint()
            newplln = i.Duplicate()
            trnsf = rg.Transform.Scale(center, scalefactor)
            newplln.Transform(trnsf)
            newplln_ = newplln.ToPolylineCurve()
            test = newplln_.Contains(self.point)
            if i.IsClosed and test == rg.PointContainment.Inside:
                newplln_ = None
                prmtr = newplln.ClosestParameter(self.point)
                cpt = newplln.ClosestPoint(self.point)
                cntpt = newplln.CenterPoint()
                vec0 = Vector2Pt(cntpt, cpt)
                vec1 = newplln.TangentAt(prmtr)
                vec1 *= 500
                vec = rg.Vector3d.Add(vec0, vec1)
                vec = VectorConstrain(vec, .75, .0)
                mult = rg.Vector3d.Multiply(self.vel, vec)
                if mult >= 0:
                    self.applyForce(vec)
                else:
                    self.applyForce(-vec)
                self.point = cpt

    def arrived(self, tol=1.5) -> bool: #10
        verif1 = ((self.destiny[0] - tol) < self.point[0] < (self.destiny[0] + tol))
        verif2 = ((self.destiny[1] - tol) < self.point[1] < (self.destiny[1] + tol))
        if verif1 and verif2:   return True
        else:                   return False
    
"""
--------------------------------------------------------------------------- Classe Sistema de Partículas
"""

@dataclass
class ParticleSystem:
    mList: list
    mListPoint: list
    timer: dict
    pathList: list
    edges: rg.Polyline()
    numPassed: float
    numNotPassed: float
    radiusPathPoint: float
    listEntries: list
    limitPath: int
    classMoverType = Mover

    def addMover(self) -> None: #1
        """Acrescenta uma partícula no sistema"""
        entrance, exit = EntranceExit(self.listEntries)
        m = self.classMoverType(
            entrance,
            exit,
            VectorRandom(),
            rg.Vector3d(0, 0, 0),
            r.randint(1, 3),
            self.edges
        )
        self.mList.append(m)
        self.mListPoint.append(m.point)

    def goToDestiny(self, vecMax, vecMin) -> None: #2
        """Aplicação do comportamento em todas as partículas"""
        for m in self.mList: m.goToDestiny(vecMax, vecMin)
    
    def flowFieldFollowing(self, listPoint, listVector, vecMax, vecMin): #3
        """Aplicação do comportamento em todas as partículas"""
        for m in self.mList: m.flowFieldFollowing(listPoint, listVector, vecMax, vecMin)

    def repelClosest(self, vecMax, vecMin) -> None: #4
        """Aplicação do comportamento em todas as partículas"""
        for m in self.mList: m.repelClosest(self.mListPoint, vecMax, vecMin)
    
    def repelFromObstacle(self, obsts) -> None: #5
        """Aplicação do comportamento em todas as partículas"""
        for m in self.mList: m.repelFromObstacle(obsts, scalefactor)
    
    def testPoint(self, listPoint, dictCoordenates, dictVazios) -> None: #6
        """Identifica o centro da face mais próximo do agente"""
        for m in self.mList:
            if m != None:
                indices = list(R.Point3dKNeighbors(listPoint, [m.point], 2))
                ind = indices[0][1]
                pt = listPoint[ind]
                dictCoordenates[(round(pt.X, 2), round(pt.Y, 2))] += 1
    
    def makePathWalker(self, time, dictCoordenates, dictVazios, listIndex): #7
        """Constrói o caminho dos agentes"""
        listToErase = []
        for i, j in dictCoordenates.items():                                            # percorre as chaves (células, i) e valores (número de passadas, j) do dicionários grid
            cpTime = copy.deepcopy(time)                                                # cópia do número do valor tempo
            cpNumP = copy.deepcopy(dictCoordenates[i])                                  # cópia do número de passadas naquele tempo
            if i not in self.timer:                                                     # verifica se a chave está na lista de timer
                if (j >= self.numPassed) and (len(self.pathList) <= self.limitPath):    # verifica se o número de passadas for maior que o valor mínimo
                    self.timer[i] = (cpTime, cpNumP)                                    # dicionário timer guarda na chave célula a tupla de valores tempo e número de passadas neste tempo
                    self.pathList.append(rg.Point3d(i[0], i[1], 0))                     # célula é adicionada a lista de caminhos
                    listIndex.append(dictVazios[i])
                else: pass
            else:
                vl = self.timer[i][1]                                                   # se a chave já está no dicionário timer, a variável vl atribui número de passadas desde a última atualização desta
                if j > vl: self.timer[i] = (cpTime, cpNumP)                             # atualização do dicionário
                else: pass
        for i, j in self.timer.items():                                                 # percorre as chaves (células, i) e valores (tupla (tempo, número de passadas), j) do dicionários timer
            if (j[0] + self.numNotPassed == time) and (dictCoordenates[i] == j[1]):     # se o tempo desde a última atualização somado ao tempo limite for igual ao tempo atual e o número de passadas não tiver sido atualizado
                listToErase.append(i)                                                   # célula é adicionada a lista para ser apagada das listas e dicionários
        for i in listToErase:
            self.timer.pop(i)                                                           # remove a célula do dicionário timer
            self.pathList.remove(rg.Point3d(i[0], i[1], 0))                             # remove a célula da lista de caminhos
            dictCoordenates[i] = 0                                                      # zera o número de passadas da célula
            listIndex.remove(dictVazios[i])
    
    def makePathConstructor(self, time, dictCoordenates, dictVazios, listIndex, listIndexWalker): #8
        """Constrói o caminho dos agentes"""
        listToErase = []
        for i, j in dictCoordenates.items():                                            # percorre as chaves (células, i) e valores (número de passadas, j) do dicionários grid
            cpTime = copy.deepcopy(time)                                                # cópia do número do valor tempo
            cpNumP = copy.deepcopy(dictCoordenates[i])                                  # cópia do número de passadas naquele tempo
            if i not in self.timer:                                                     # verifica se a chave está na lista de timer
                if (j >= self.numPassed) and (len(self.pathList) <= self.limitPath):    # verifica se o número de passadas for maior que o valor mínimo
                    self.timer[i] = (cpTime, cpNumP)                                    # dicionário timer guarda na chave célula a tupla de valores tempo e número de passadas neste tempo
                    self.pathList.append(rg.Point3d(i[0], i[1], 0))                     # célula é adicionada a lista de caminhos
                    listIndex.append(dictVazios[i])
                else: pass
            else:
                vl = self.timer[i][1]                                                   # se a chave já está no dicionário timer, a variável vl atribui número de passadas desde a última atualização desta
                if j > vl: self.timer[i] = (cpTime, cpNumP)                             # atualização do dicionário
                else: pass
        for i, j in self.timer.items():                                                 # percorre as chaves (células, i) e valores (tupla (tempo, número de passadas), j) do dicionários timer
            if (j[0] + self.numNotPassed == time) and (dictCoordenates[i] == j[1]):     # se o tempo desde a última atualização somado ao tempo limite for igual ao tempo atual e o número de passadas não tiver sido atualizado
                listToErase.append(i)                                                   # célula é adicionada a lista para ser apagada das listas e dicionários
            elif dictVazios[i] in listIndexWalker:
                listToErase.append(i)
            else: pass
        for i in listToErase:
            self.timer.pop(i)                                                           # remove a célula do dicionário timer
            self.pathList.remove(rg.Point3d(i[0], i[1], 0))                             # remove a célula da lista de caminhos
            dictCoordenates[i] = 0                                                      # zera o número de passadas da célula
            listIndex.remove(dictVazios[i])

    def attractToPath(self, listPoint, vecMax, vecMin, tol) -> None: #9
        """aplicação do comportamento de preferir passar pelo caminho"""
        for m in self.mList:
            m.attractToPath(self.pathList, vecMax, vecMin, tol)
    
    def repelPath(self, listPoint, vecMax, vecMin, tol) -> None: #10
        """aplicação do comportamento de preferir passar pelo caminho"""
        for m in self.mList:
            m.attractToPath(self.pathList, vecMax, vecMin, tol)

    def drawPath(self) -> list: #11
        areaMin = meshFaceDim * 1.25
        lenMax = meshFaceDim * 1.5
        if len(self.pathList) > 10:
            wallTriangList = Delaunay(self.pathList, areaMin, lenMax)
            return MergePolylines(wallTriangList, .1)
        else:
            pass

    def applyForce(self, force) -> None: #12
        """Aplicação de força em todas as partículas"""
        for m in self.mList: m.applyForce(force)

    def update(self, listPoint, dictCoordenates, dictVazios) -> None: #13
        """Atualiza a posição de todas as partículas"""
        for i, m in enumerate(self.mList):
            if m.arrived():
                self.mList[i] = None
                self.addMover()
            else:
                m.update()
        self.testPoint(listPoint, dictCoordenates, dictVazios)
        self.mList = list(filter(lambda x: (x != None), self.mList))    # remove todas as partículas com o valor None

    def getPoint(self) -> list: #14
        """Retorna a lista de pontos com as posições das partículas em uma determinada iteração"""
        pointList = []
        for m in self.mList: pointList.append(m.point)
        return pointList

    def getVelocity(self) -> list: #15
        """Retorna a lista de vetores velocidade em uma determinada iteração"""
        velList = []
        for m in self.mList: velList.append(m.vel)
        return velList


"""
--------------------------------------------------------------------------- Classe Caminhantes do tipo Mover
"""

class Walker(Mover):
    def __init__(self, point, destiny, vel, acc, mass, edges):
        super().__init__(point, destiny, vel, acc, mass, edges)


"""
--------------------------------------------------------------------------- Classe Ciclistas do tipo Mover
"""


class Constructor(Mover):
    def __init__(self, point, destiny, vel, acc, mass, edges):
        super().__init__(point, destiny, vel, acc, mass, edges)


"""
--------------------------------------------------------------------------- Classe PS Caminhantes do tipo Particle System
"""

class PSWalker(ParticleSystem):
    def __init__(self, wlkList, wlkListPts, wlkTimer, wlkPathPts, edges, numPassed, numNotPassed, radiusPathPoint, listEntries, limitPath):
        super().__init__(wlkList, wlkListPts, wlkTimer, wlkPathPts, edges, numPassed, numNotPassed, radiusPathPoint, listEntries, limitPath)
        self.classMoverType = Walker


"""
--------------------------------------------------------------------------- Classe PS Caminhantes do tipo Particle System
"""

class PSConstructor(ParticleSystem):
    def __init__(self, cnstList, cnstListPts, cnstTimer, cnstPathPts, edges, numPassed, numNotPassed, radiusPathPoint, listEntries, limitPath):
        super().__init__(cnstList, cnstListPts, cnstTimer, cnstPathPts, edges, numPassed, numNotPassed, radiusPathPoint, listEntries, limitPath)
        self.classMoverType = Constructor


"""
--------------------------------------------------------------------------- Tempo
"""


meshFaceDim = LandscapeParam[0]
heightTopo = LandscapeParam[1]
distMinTopo = LandscapeParam[2]
freqModTopo = LandscapeParam[3]
timeMinModTopo = LandscapeParam[4]
seed = LandscapeParam[5]

numWalkers = WalkersParam[0]
numWlkPassedOn = WalkersParam[1]
numWlkNotPassedOn = WalkersParam[2]
maxWlkPathPoints = WalkersParam[3]

numConstructors = ConstructorsParam[0]
numCnstPassedOn = ConstructorsParam[1]
numCnstNotPassedOn = ConstructorsParam[2]
maxCnstPathPoints = ConstructorsParam[3]

P0 = (BezierParam[0][0], BezierParam[0][1])
T0 = (BezierParam[1][0], BezierParam[1][1])
P1 = (BezierParam[2][0], BezierParam[2][1])
T1 = (BezierParam[3][0], BezierParam[3][1])


radiusPathPoint = 5
tolPath = 2
scalefactor = 1.5

if reset:
    r.seed(seed)
    t_ = 0

    ptsMesh, vcsMesh, fcsMesh, coordMesh, vaziosMesh, newMesh = MeshFlowField(mesh, t_, meshFaceDim)
    lnsMesh, crvMesh = getEntries(newMesh, entries)
    originalMesh = newMesh.Duplicate()
    WcoordMesh = copy.deepcopy(coordMesh)
    WvaziosMesh = copy.deepcopy(vaziosMesh)
    CcoordMesh = copy.deepcopy(coordMesh)
    CvaziosMesh = copy.deepcopy(vaziosMesh)

    WpointList, WvelList, WpathPts, WpathCrvs, WindexList = [], [], [], [], []
    CpointList, CvelList, CpathPts, CpathCrvs, CindexList = [], [], [], [], []

    wlkList, wlkListPts, wlkTimer, wlkPathPts, wlkPathCrvs = [], [], {}, [], []
    cnstList, cnstListPts, cnstTimer, cnstPathPts, cnstPathCrvs = [], [], {}, [], []

    psw = PSWalker(
        wlkList,
        wlkListPts,
        wlkTimer,
        wlkPathPts,
        crvMesh,
        numWlkPassedOn,
        numWlkNotPassedOn,
        radiusPathPoint,
        lnsMesh,
        maxWlkPathPoints)
    
    psc = PSConstructor(
        cnstList,
        cnstListPts,
        cnstTimer,
        cnstPathPts,
        crvMesh,
        numCnstPassedOn,
        numCnstNotPassedOn,
        radiusPathPoint,
        lnsMesh,
        maxCnstPathPoints)

    for i in range(numWalkers): psw.addMover()
    for i in range(numConstructors): psc.addMover()

    WpointList = psw.getPoint()
    WvelList = psw.getVelocity()
    CpointList = psc.getPoint()
    CvelList = psc.getVelocity()

else:
    t_ += 1

    psw.repelClosest(.9, 0)
    psw.goToDestiny(1, .91)
    psw.flowFieldFollowing(ptsMesh, vcsMesh, .9, 0)
    psw.makePathWalker(t_, WcoordMesh, WvaziosMesh, WindexList)
    psw.attractToPath(WpathPts, .9, .75, tolPath)
    #psw.repelFromObstacle(obstacles)
    psw.update(ptsMesh, WcoordMesh, WvaziosMesh)

    psc.repelClosest(.75, 0)
    psc.goToDestiny(1, .91)
    psc.flowFieldFollowing(ptsMesh, vcsMesh, .75, 0)
    psc.makePathConstructor(t_, CcoordMesh, CvaziosMesh, CindexList, WindexList)
    psc.repelPath(WpathPts, .9, .75, tolPath)
    psc.attractToPath(CpathPts, .9, .75, tolPath)
    #psc.repelFromObstacle(obstacles)
    psc.update(ptsMesh, CcoordMesh, CvaziosMesh)

    WpointList = psw.getPoint()
    WvelList = psw.getVelocity()
    WpathPts = psw.pathList

    CpointList = psc.getPoint()
    CvelList = psc.getVelocity()
    CpathPts = psc.pathList

    if t_ % 20 == 0:
        WpathCrvs = psw.drawPath()
        CpathCrvs = psc.drawPath()

    if t_ % freqModTopo == 0 and t_ > timeMinModTopo and CpathCrvs != None:
        meshTopo = MeshNewTopology(newMesh, originalMesh, CpathCrvs, heightTopo)
        ptsMesh, vcsMesh, fcsMesh, coordNewMesh, vaziosNewMesh, newMesh = MeshFlowField(meshTopo, t_, meshFaceDim)
        for i, j in coordMesh.items():
            coordNewMesh[i] = j
        for i, j in vaziosMesh.items():
            vaziosNewMesh[i] = j
        WcoordMesh = coordNewMesh
        CcoordMesh = coordNewMesh
        WvaziosMesh = vaziosNewMesh
        CcoordMesh = coordNewMesh

# output
branch = t_
ptsWalkers = WpointList
vecsWalkers = WvelList
pathPtsWalkers = WpathPts
pathCrvsWalkers = WpathCrvs
ptsConstructors = CpointList
vecsConstructors = CvelList
pathPtsConstructors = CpathPts
pathCrvsConstructors = CpathCrvs
mesh = newMesh
