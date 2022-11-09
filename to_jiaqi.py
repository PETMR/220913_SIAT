import os
import numpy as np
import point_process as pp
from skimage import io

pathi1 = '/home/rex/eight/SIAT-3DFE_points'
pathi2 = '/home/rex/eight/SIAT-3DFE_keypoints'
patho = '/home/rex/eight/SIAT-jiaqi'
# patho = '/home/rex/eight/temp'
points = os.listdir(pathi1)
points.sort()

for i in range(0, len(points)):
# for i in points:

    pointsDir = os.path.join(pathi1, points[i])
    baseName = points[i].split('_')[0]+'_'+points[i].split('_')[1]+'_'+points[i].split('_')[2]
    keyPointsFile = baseName+'_'+'keypoints.txt'
    keyPointsDir = os.path.join(pathi2, keyPointsFile)
    inkp = pp.readXYZ(keyPointsDir)
    inkp = inkp[:, 17:]
    rotate, trans = pp.forwardFace(inkp)
    xyz, rgb = pp.readXYZRGB(pointsDir)
    xyz = (rotate.dot(xyz) - trans)

    # out = pp.render_colors(xyz, rgb, 512, 512)
    # io.imsave(os.path.join(patho, 'rgb', baseName + '_' + '0' + "_" + '0' + '.jpg'), out[1])
    print(points[i])
    for rx in range(-5, 6, 5):
        for ry in range(-10, 11, 10):
            angles = np.zeros(3)
            angles[0] = rx * np.pi / 180
            angles[1] = ry * np.pi / 180
            matrix = pp.angle2matrix(angles)
            tmpxyz = matrix.dot(xyz)
            out = pp.render_colors(tmpxyz, rgb, 512, 512)
            jpg = out[1].astype(np.uint8)
            io.imsave(os.path.join(patho, 'rgb', baseName + '_' + str(rx) + "_" + str(ry) + '.jpg'), jpg)
            # io.imsave(os.path.join(patho, 'xyz', baseName + '_' + str(rx) + "_" + str(ry) + '.png'), out[0])
            np.save(os.path.join(patho, 'xyz', baseName + '_' + str(rx) + "_" + str(ry) + '.npy'), out[0])
            pass

pass
