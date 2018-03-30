'''
Created on Mar 6, 2018
This is an implementation of Characterizing Driving Styles with Deep Learning 2016: Generating Feature Matrix
@author: sobhan
'''
import cPickle
import numpy as np
import math
from scipy import stats
import time

class point:
    lat = 0
    lng = 0
    time = 0
    def __init__(self, time, lat, lng):
        self.lat = lat
        self.lng = lng
        self.time = time

class basicFeature:
    speedNorm = 0
    diffSpeedNorm = 0
    accelNorm = 0
    diffAccelNorm = 0
    angularSpeed = 0
    def __init__(self, speedNorm, diffSpeedNorm, accelNorm, diffAccelNorm, angularSpeed):
        self.speedNorm = speedNorm
        self.diffSpeedNorm= diffSpeedNorm
        self.accelNorm= accelNorm
        self.diffAccelNorm = diffAccelNorm
        self.angularSpeed = angularSpeed

def haversineDistance(fLat, fLon, sLat, sLon, km=False):
        fLat = np.radians(float(fLat))
        fLon = np.radians(float(fLon))
        sLat = np.radians(float(sLat))
        sLon = np.radians(float(sLon))
        
        R = 6371000.0 #meters
        if km:
            R = 6371.0 #KM: earth/ radius
        dLon = sLon - fLon
        dLat = sLat - fLat
        a = pow(np.sin(dLat/2.0), 2) + (np.cos(fLat) * np.cos(sLat) * pow(np.sin(dLon/2.0), 2))
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        
        distance = R * c 
        return distance
    
def returnAngularDisplacement(fLat, fLon, sLat, sLon):
    #Inspired by: https://www.quora.com/How-do-I-convert-radians-per-second-to-meters-per-second
    
    fLat = np.radians(float(fLat))
    fLon = np.radians(float(fLon))
    sLat = np.radians(float(sLat))
    sLon = np.radians(float(sLon))
    
    dis = np.sqrt((fLat-sLat)**2 + (fLon-sLon)**2)
    return dis

def generateStatisticalFeatureMatrix(Ls=256, Lf=4):
    #load trajectories
    trajectories = {}
    filename = 'smallSample_5_5.csv'
    with open('Samples/'+filename, 'r') as f:
        lines = f.readlines()
        ct = ""
        tj = []
        for ln in lines:
            pts = ln.replace('\r\n','').split(',')
            if pts[1] != ct:
                if ct == "" and pts[0]=="Driver":
                    continue
                if len(tj) >0:
                    trajectories[ct] = tj
                tj = []
                tj.append(point(int(pts[2]), float(pts[3]), float(pts[4])))                
                ct = pts[1]
            else:
                tj.append(point(int(pts[2]), float(pts[3]), float(pts[4]))) 
        trajectories[ct] = tj
#     cPickle.dump(trajectories, open('trajectories', 'w'))
    print('Raw Trajectory Data is loaded! |Trajectories|:' + str(len(trajectories)))
    #Generate Basic Features for each trajectory
    basicFeatures = {}
    for t in trajectories:
        points = trajectories[t]
        traj = []
        lastSpeedNorm = lastAccelNorm = -1
        lastLatSpeed = lastLngSpeed = 0
        for i in range(1, len(points)):
#             surfDisplacement = haversineDistance(points[i-1].lat, points[i-1].lng, points[i].lat, points[i].lng)
            speedNorm = np.sqrt((points[i].lat-points[i-1].lat)**2 + (points[i].lng-points[i-1].lng)**2) #Time difference is unit
            diffSpeedNorm = 0
            if lastSpeedNorm > -1:
                diffSpeedNorm = np.abs(speedNorm - lastSpeedNorm)
                
            latSpeed = np.abs(points[i].lat-points[i-1].lat)
            lngSpeed = np.abs(points[i].lng-points[i-1].lng)
            accelNorm = np.sqrt((latSpeed - lastLatSpeed)**2 + (lngSpeed - lastLngSpeed)**2) #Time difference is unit
            
            diffAccelNorm = 0
            if lastAccelNorm > -1:
                diffAccelNorm = np.abs(accelNorm - lastAccelNorm)
            
            angularSpeed = returnAngularDisplacement(points[i-1].lat, points[i-1].lng, points[i].lat, points[i].lng)
             
            lastSpeedNorm = speedNorm
            lastAccelNorm = accelNorm
            lastLatSpeed = latSpeed
            lastLngSpeed = lngSpeed
            
            traj.append([speedNorm, diffSpeedNorm, accelNorm, diffAccelNorm, angularSpeed])        
        basicFeatures[t] = traj
        
    print('Basic Features are created!')
    
    #Generate Statistical Feature Matrix
    start = time.time()
    statisticalFeatureMatrix = {}
    for t in basicFeatures:
        print 'processing', t      
        matricesForTrajectory = []
        traj= basicFeatures[t]
        ranges = returnSegmentIndexes(Ls, len(traj))        
        for p in ranges:
            matrixForSegment = []
            st = p[0]
            while st < p[1]:
                en = min(st+Lf, p[1])
                column = []     
                for fIdx in range(5):
                    arr = []
                    mean = 0.0
                    for i in range(st, en):            
                        mean += traj[i][fIdx]
                        arr.append(traj[i][fIdx])      
                    arr.sort()
                    mean = mean/len(arr)
                    column.append(mean) #mean
                    column.append(arr[0]) #min
                    column.append(arr[len(arr)-1]) #max                    
                    column.append(stats.scoreatpercentile(arr, 25)) #25% percentile
                    if len(arr)%2 == 0:
                        column.append((arr[len(arr)/2] + arr[(len(arr)/2) -1])/2.0) #50% percentile
                    else:
                        column.append(arr[len(arr)/2]) #50% percentile
                    column.append(stats.scoreatpercentile(arr, 75)) #75% percentile
                    std = 0
                    for a in arr:
                        std += (a-mean)**2
                    column.append(math.sqrt(std)) #standard deviation
                matrixForSegment.append(column)
                st += Lf/2            
            matricesForTrajectory.append(matrixForSegment)
              
        statisticalFeatureMatrix[t] = matricesForTrajectory
    
    print 'elpased time:', time.time()-start
    normalizedStatFeatureMatrix = normalizeStatFeatureMatrix(statisticalFeatureMatrix, minimum=0, maximum=40)
    print 'Normalization is completed!'
    cPickle.dump(normalizedStatFeatureMatrix, open('data/'+filename.replace('.csv', ''), 'wb'))
    #normalizedStatFeatureMatrix: In this dictionary, we have an array of feature matrices for each trajectory. 
    # Each feature matrix has 35 columns and up to 128 rows. Usually, the last stat feature matrix of a trajectory has less than 128 rows. 
            
def returnSegmentIndexes(Ls, leng):
    ranges = []
    start = 0
    while True:        
        end = min(start+Ls, leng-1)
        ranges.append([start, end])
        start += Ls/2
        if end == leng-1:
            break        
    return ranges

def normalizeStatFeatureMatrix(statisticalFeatureMatrix, minimum=0, maximum=40):
    normalizedStatFeatureMatrix = {}
    
    for t in statisticalFeatureMatrix:
        matricesForTrajectory = statisticalFeatureMatrix[t]
        mins = np.ones(35)*10000
        maxs = np.ones(35)*-10000        
        for m in matricesForTrajectory:
            n = np.asarray(m)            
            for i in range(35):
                mins[i] = min(mins[i], min(n[:,i]))
                maxs[i] = max(maxs[i], max(n[:,i]))
        normalizedMatrices = []
        for m in matricesForTrajectory:
            _m = []
            for i in range(len(m)):
                row = []
                for j in range(35):     
                    if mins[j] == maxs[j] == 0:
                        row.append(0.0)
                    elif mins[j] == maxs[j]:
                        row.append(maximum)
                    else:
                        val = (m[i][j]-mins[j])/(maxs[j] - mins[j])                                                                    
                        row.append((maximum-minimum)*val + minimum)                
                _m.append(row)
            normalizedMatrices.append(_m)
        normalizedStatFeatureMatrix[t] = normalizedMatrices
        
    return normalizedStatFeatureMatrix
    
if __name__ == '__main__':
    generateStatisticalFeatureMatrix()    
