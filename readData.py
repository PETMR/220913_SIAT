import os.path
import struct
import numpy as np


def readModel(path):
    pathCie = os.path.join(path, 'model.cie')
    pathRgb = os.path.join(path, 'model.jpg')
    os.path.exists(pathCie)
    os.path.exists(pathRgb)
    if (os.path.exists(pathCie) and os.path.exists(pathRgb)) == False:
        return None

    # pathCie = os.path.join(path, 'model.cie')
    f = open(pathCie, 'rb+')
    bytestring = f.read(16)  # 所要读取的信息为4字节Int
    length = struct.unpack("4I", bytestring)
    if length == 0: return None

    vert_xyz = np.zeros([3, length[0]], dtype=float)
    vertnorm = np.zeros([3, length[0]], dtype=float)
    verttext = np.zeros([8, length[0]], dtype=float)

    for i in range(length[0]):
        bytestring = f.read(64)  #
        tmp = struct.unpack("3fc3fc8f", bytestring)
        vert_xyz[0, i] = tmp[0]
        vert_xyz[1, i] = tmp[1]
        vert_xyz[2, i] = tmp[2]

        vertnorm[0, i] = tmp[4]
        vertnorm[1, i] = tmp[5]
        vertnorm[2, i] = tmp[6]

        verttext[0, i] = tmp[8]
        verttext[1, i] = tmp[9]
        verttext[2, i] = tmp[10]
        verttext[3, i] = tmp[11]
        verttext[4, i] = tmp[12]
        verttext[5, i] = tmp[13]
        verttext[6, i] = tmp[14]
        verttext[7, i] = tmp[15]

    faces = np.zeros([4, length[1]], dtype=int)
    bytestring = f.read(4 * length[1] * 4)  #
    tmp = struct.unpack(str(length[1] * 4) + "i", bytestring)

    for i in range(length[2]):
        faces[0, i] = tmp[i * 4 + 0]
        faces[1, i] = tmp[i * 4 + 1]
        faces[2, i] = tmp[i * 4 + 2]
        faces[3, i] = tmp[i * 4 + 3]

        # f.seek(4 * tt00[1] * 4, 1)

    facesEx = np.zeros([4, length[2]], dtype=np.float32)
    bytestring = f.read(4 * length[2] * 4)  #
    tmp = struct.unpack(str(length[2] * 4) + "f", bytestring)
    for i in range(length[2]):
        facesEx[0, i] = tmp[i * 4 + 0]
        facesEx[1, i] = tmp[i * 4 + 1]
        facesEx[2, i] = tmp[i * 4 + 2]
        facesEx[3, i] = tmp[i * 4 + 3]

    fkp = np.zeros(length[3], dtype=int)
    bytestring = f.read(4 * length[3])  #
    tmp = struct.unpack(str(length[3]) + "i", bytestring)
    fkpXYZ = np.zeros([3, length[3]], dtype=float)
    for i in range(length[3]):
        fkp[i] = tmp[i]
        fkpXYZ[0, i] = vert_xyz[0, fkp[i]]
        fkpXYZ[1, i] = vert_xyz[1, fkp[i]]
        fkpXYZ[2, i] = vert_xyz[2, fkp[i]]

    import cv2
    # pathRgb = os.path.join(path, 'model.jpg')
    img = cv2.imread(pathRgb)
    img = img.transpose((1, 0, 2))

    vert_rgb = np.zeros([3, length[0]], dtype=int)
    vert_Ex = np.zeros([2, length[0]], dtype=float)
    for i in range(facesEx.shape[1]):
        index = faces[0, i]
        if vert_Ex[0, index] == 0 and vert_Ex[1, index] == 0:
            vert_Ex[0, index] = facesEx[0, i]
            vert_Ex[1, index] = facesEx[1, i]

    for i in range(vert_rgb.shape[1]):
        tmp = verttext[:, i].copy()
        if tmp[0] < 0 or tmp[0] > 1: continue
        if tmp[1] < 0 or tmp[1] > 1: continue
        if tmp[2] < 0 or tmp[2] > 1: continue
        if tmp[3] < 0 or tmp[3] > 1: continue
        tmp[0] = img.shape[0] * tmp[0]
        tmp[1] = img.shape[1] * tmp[1]
        tmp[2] = img.shape[0] * tmp[2]
        tmp[3] = img.shape[1] * tmp[3]
        tmp = tmp.astype(int)
        if (vert_Ex[0, i] == 0 and vert_Ex[1, i] == 0):
            continue
        a = vert_Ex[0, i] / (vert_Ex[0, i] + vert_Ex[1, i])
        b = 1 - a

        vert_rgb[:, i] = np.minimum(np.maximum((a * img[tmp[0], tmp[1], :] + b * img[tmp[2], tmp[3], :]), 0),
                                    255).astype(int)

    return vert_xyz, vert_rgb, fkpXYZ


def writeXYZRGB(xyz, rgb, path):
    with open(path, 'w+', encoding='utf-8') as f:
        for i in range(xyz.shape[1]):
            if (rgb[2, i] == 0 and rgb[1, i] == 0 and rgb[0, i] == 0): continue
            sstr = str(format(xyz[0, i], '.2f')) + ' ' + \
                   str(format(xyz[1, i], '.2f')) + ' ' + \
                   str(format(xyz[2, i], '.2f')) + ' ' + \
                   str(format(rgb[2, i], 'd')) + ' ' + \
                   str(format(rgb[1, i], 'd')) + ' ' + \
                   str(format(rgb[0, i], 'd')) + '\n'
            f.write(sstr)

def writeXYZ(xyz, path):
    with open(path, 'w+', encoding='utf-8') as f:
        for i in range(xyz.shape[1]):
            sstr = str(format(xyz[0, i], '.2f')) + ' ' + \
                   str(format(xyz[1, i], '.2f')) + ' ' + \
                   str(format(xyz[2, i], '.2f')) + '\n'
            f.write(sstr)
