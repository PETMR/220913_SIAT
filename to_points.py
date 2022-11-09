"/home/rex/eight/SIAT-3DFE/male/1/0/model.cie"

import os
import readData

pathi = '/home/rex/eight/SIAT-3DFE/male'
patho1 = '/home/rex/eight/SIAT-3DFE_points'
patho2 = '/home/rex/eight/SIAT-3DFE_keypoints'

for g in range(100000):
    person = os.listdir(pathi)
    # for i in range(len(person)):
    #     person[i] = person[i].zfill(4)
    person.sort()
    for i in person:
        # i = person[204]
        expression = os.listdir(os.path.join(pathi, i))
        for j in expression:
            # j = expression[10]
            path = os.path.join(pathi, i, j)
            out = readData.readModel(path)
            if out is None:
                print(path)
            else:

                readData.writeXYZRGB(out[0], out[1], os.path.join(patho1, '1_' + str(i).zfill(4) + '_' + str(j).zfill(2) + '_points.txt'))
                readData.writeXYZ(out[2], os.path.join(patho2, '1_' + str(i).zfill(4) + '_' + str(j).zfill(2) + '_keypoints.txt'))
    break









