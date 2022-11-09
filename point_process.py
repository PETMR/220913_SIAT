import numpy as np
import math

def render_colors(inXYZ, inRGB, h, w):
    assert inXYZ.shape[1] == inRGB.shape[1]

    # inXYZ = inXYZ * (w - 12) / (np.max(inXYZ[1]) - np.min(inXYZ[1])) + np.array([int(h / 2), int(w / 2), 0]).reshape([3, -1])
    inXYZ = inXYZ * 1 + np.array([int(h / 2), int(w / 2), 0]).reshape([3, -1])
    inXYZ[2] = -inXYZ[2]

    depthBuffer = np.zeros([h, w], dtype=float) - 9999.
    outXYZ = np.zeros([h, w, 3], dtype=float)
    outRGB = np.zeros([h, w, 3], dtype=int)

    for i in range(inXYZ.shape[1]):
        v = int(inXYZ[1, i])
        u = int(inXYZ[0, i])
        z = inXYZ[2, i]
        if u >= w or u < 0: continue
        if v >= h or v < 0: continue
        if depthBuffer[v, u] > z: continue
        outXYZ[v, u, :] = inXYZ[:, i]
        outRGB[v, u, :] = inRGB[:, i]
        depthBuffer[v, u] = z
        pass

    for i in range(4):
        for v in range(3, h - 4):
            for u in range(3, w - 4):
                if depthBuffer[v, u] != -9999.: continue
                if depthBuffer[v, u - 1] != -9999. and depthBuffer[v, u + 1] != -9999.:
                    depthBuffer[v, u] = (depthBuffer[v, u - 1] + depthBuffer[v, u + 1]) / 2
                    outXYZ[v, u, :] = (outXYZ[v, u - 1, :] + outXYZ[v, u + 1, :]) / 2
                    outRGB[v, u, :] = (outRGB[v, u - 1, :] + outRGB[v, u + 1, :]) / 2
                    pass
        for v in range(3, h - 4):
            for u in range(3, w - 4):
                if depthBuffer[v, u] != -9999.: continue
                if depthBuffer[v - 1, u] != -9999. and depthBuffer[v + 1, u] != -9999.:
                    depthBuffer[v, u] = (depthBuffer[v + 1, u] + depthBuffer[v, u + 1]) / 2
                    outXYZ[v, u, :] = (outXYZ[v - 1, u, :] + outXYZ[v + 1, u, :]) / 2
                    outRGB[v, u, :] = (outRGB[v - 1, u, :] + outRGB[v + 1, u, :]) / 2
                    pass

    return outXYZ, outRGB


def angle2matrix(angles):
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    x, y, z = angles[0], angles[1], angles[2]
    # x
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(x), -math.sin(x)],
                   [0, math.sin(x), math.cos(x)]])
    # y
    Ry = np.array([[math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]])
    # z
    Rz = np.array([[math.cos(z), -math.sin(z), 0],
                   [math.sin(z), math.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R


def computePointNormal(xyz):
    out = np.zeros(3)
    out2 = np.zeros(3)
    mu = np.zeros(3)
    for i in range(xyz.shape[1]):
        mu = mu + xyz[:, i]
    mu = mu / xyz.shape[1]
    Q = np.transpose(xyz) - mu.reshape((-1, 3))
    QT = np.transpose(Q)
    H = np.dot(QT, Q)
    u1, s1, v1 = np.linalg.svd(np.transpose(H))
    if (u1[2, 2] < 0):
        out[0] = -u1[0, 2]
        out[1] = -u1[1, 2]
        out[2] = -u1[2, 2]
    else:
        out[0] = u1[0, 2]
        out[1] = u1[1, 2]
        out[2] = u1[2, 2]

    if (u1[1, 1] < 0):
        out2[0] = -u1[0, 1]
        out2[1] = -u1[1, 1]
        out2[2] = -u1[2, 1]
    else:
        out2[0] = u1[0, 1]
        out2[1] = u1[1, 1]
        out2[2] = u1[2, 1]

    return out, out2

def readXYZRGB(path):
    fd = open(path, "r")
    lines = fd.readlines()
    xyz = np.zeros((3, len(lines)), dtype=float)
    rgb = np.zeros((3, len(lines)), dtype=int)
    for i in range(len(lines)):
        tmp = lines[i].strip().split(' ')
        xyz[0, i] = float(tmp[0])
        xyz[1, i] = float(tmp[1])
        xyz[2, i] = float(tmp[2])
        rgb[0, i] = int(tmp[3])
        rgb[1, i] = int(tmp[4])
        rgb[2, i] = int(tmp[5])
    fd.close()
    return xyz, rgb

def readXYZ(path):
    fd = open(path, "r")
    lines = fd.readlines()
    inkp = np.zeros((3, len(lines)), dtype=float)
    for i in range(len(lines)):
        tmp = lines[i].strip().split(' ')
        inkp[0, i] = float(tmp[0])
        inkp[1, i] = float(tmp[1])
        inkp[2, i] = float(tmp[2])
    fd.close()
    return inkp

# def forwardFace(inkp):
#     normal, normal2 = computePointNormal(inkp)
#     angles = np.zeros(3)
#     angles[0] = np.arctan(normal[1] / normal[2])
#     angles[1] = -np.arcsin(normal[0])
#     angles[2] = 0
#     matrix1 = angle2matrix(angles)
#     inkp2 = matrix1.dot(inkp)
#     nn, nn2 = computePointNormal(inkp2)
#     angles[0] = 0
#     angles[1] = 0
#     angles[2] = np.arcsin(nn2[1])
#     matrix2 = angle2matrix(angles)
#     inkp3 = matrix2.dot(inkp2)
#     # nf, nf2 = computePointNormal(inkp3)
#     rotate = matrix2.dot(matrix1)
#     inkp4 = np.delete(inkp3, 51, 1)
#     trans = (inkp4.sum(1) / inkp4.shape[1]).reshape([3, -1])
#
#     return rotate, trans

def forwardFace(inkp):
    inkp = np.delete(inkp, -2, 1)
    normal, normal2 = computePointNormal(inkp)
    angles = np.zeros(3)
    angles[0] = np.arctan(normal[1] / normal[2])
    angles[1] = -np.arcsin(normal[0])
    angles[2] = 0
    matrix1 = angle2matrix(angles)
    inkp2 = matrix1.dot(inkp)
    nn, nn2 = computePointNormal(inkp2)
    angles[0] = 0
    angles[1] = 0
    angles[2] = np.arcsin(nn2[1])
    matrix2 = angle2matrix(angles)
    inkp3 = matrix2.dot(inkp2)
    nf, nf2 = computePointNormal(inkp3)
    if abs(nf2[0] - 1) > 0.0001:
        angles[0] = 0
        angles[1] = 0
        angles[2] = np.pi - np.arcsin(nn2[1])
        matrix2 = angle2matrix(angles)
        inkp3 = matrix2.dot(inkp2)
        # nf, nf2 = computePointNormal(inkp3)

    rotate = matrix2.dot(matrix1)
    trans = (inkp3.sum(1) / inkp3.shape[1]).reshape([3, -1])

    return rotate, trans









