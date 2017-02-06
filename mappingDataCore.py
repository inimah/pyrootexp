import os
import sys
import pandas as pd
import numpy as np
import math
import random
import ROOT
import root_pandas as rpd
import root_numpy as rnp
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from scipy.spatial import distance



# function to create prob. density plot	
def createDensityPlot(data):
	#data.plot(kind='density', subplots=True, layout=(2,5), figsize=(20,5),sharex=False, sharey=False)
        data.plot(kind='density', subplots=True, layout=(6,6), figsize=(30,30),sharex=False, sharey=False)
	#plt.show()
	plt.savefig('/home/tita/PyRoot/plots/densityPlot.png')
	return 0


# function to create correlation matrix plot
def createCorrelationMatrixPlot(data):
       sns.set(style="white")
       # Pearsson correlation coefficient
       correlations = data.corr()
       f, ax = plt.subplots(figsize=(12, 10))
       # Generate a mask for the upper triangle
       mask = np.zeros_like(correlations, dtype=np.bool)
       mask[np.triu_indices_from(mask)] = True
       sns.heatmap(correlations, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5, ax=ax)
       plt.yticks(rotation=0)
       plt.xticks(rotation=90)
       plt.savefig('/home/tita/PyRoot/plots/correlationMatrix.png')
       
       return correlations


# function to create scatterplot matrix
def createScatterPlot(data):
	scatter_matrix(data)
	#plt.show()
	plt.savefig('/home/tita/PyRoot/plots/scatterPlot.png')
	return 0


def pairedPlots(df):
        sns.set(style="white")
        g = sns.PairGrid(df, diag_sharey=False)
        g.map_lower(sns.kdeplot, cmap="RdYlGn_r")
        g.map_upper(plt.scatter)
        g.map_diag(sns.kdeplot, lw=3)
        plt.savefig('/home/tita/PyRoot/plots/pairedPlots.png')

def pairedScatterplots(df):
        sns.set(style="white")
        sns.pairplot(df, hue="class", palette="husl")
        plt.savefig('/home/tita/PyRoot/plots/pairedScatters.png')
        return 0

#clear memory used of cluster engines
def clear_clients(rc):
    
    rc.results.clear()
    rc.client.results.clear()
    #rc.metadata.clear()
    rc.client.metadata.clear()
    rc.history=[]
    rc.client.history=[]
    #rc.session.digest_history.clear() 
    rc.client.session.digest_history.clear()
   

#mapping continuous values of class attributes to discrete (class region)
def mappingClasses(df):
        
        df['drGrp']=df['dr'].apply(lambda x: math.ceil(x) if x > 0 else math.floor(x))
        df['dzGrp']=df['dz'].apply(lambda x: math.ceil(x) if x > 0 else math.floor(x))
        df['dphiGrp']=df['dphir'].apply(lambda x: math.ceil(x) if x > 0 else math.floor(x))
        
        #grouping data into class region (merge positive - negative values into one region interval (e.g. 1-40)
        #by rounding up numerical values
        #any values out of this region (x > 20) --> is assigned class "41" ; (x < -20) --> is assigned class "42"
        df['drZero']=df['drGrp'].apply(lambda x: 1 if x == 0 else 0)
        df['drPos']=df['drGrp'].apply(lambda x: int(x) if x > 0 and x <= 20 else 0)
        df['drNeg']=df['drGrp'].apply(lambda x: np.abs(x)+20 if x < 0 and x >= -20 else 0)
        df['drPosPos']=df['drGrp'].apply(lambda x: int(41) if x > 20 else 0)
        df['drNegNeg']=df['drGrp'].apply(lambda x: int(42) if x < -20 else 0)
        df['drClass']=df['drZero'] + df['drPos'] + df['drNeg'] + df['drPosPos'] + df['drNegNeg']

        df['dzZero']=df['dzGrp'].apply(lambda x: 1 if x == 0 else 0)
        df['dzPos']=df['dzGrp'].apply(lambda x: int(x) if x > 0 and x <= 20 else 0)
        df['dzNeg']=df['dzGrp'].apply(lambda x: np.abs(x)+20 if x < 0 and x >= -20 else 0)
        df['dzPosPos']=df['dzGrp'].apply(lambda x: int(41) if x > 20 else 0)
        df['dzNegNeg']=df['dzGrp'].apply(lambda x: int(42) if x < -20 else 0)
        df['dzClass']=df['dzZero'] + df['dzPos'] + df['dzNeg'] + df['dzPosPos'] + df['dzNegNeg']

        df['dphiZero']=df['dphiGrp'].apply(lambda x: 1 if x == 0 else 0)
        df['dphiPos']=df['dphiGrp'].apply(lambda x: int(x) if x > 0 and x <= 20 else 0)
        df['dphiNeg']=df['dphiGrp'].apply(lambda x: np.abs(x)+20 if x < 0 and x >= -20 else 0)
        df['dphiPosPos']=df['dphiGrp'].apply(lambda x: int(41) if x > 20 else 0)
        df['dphiNegNeg']=df['dphiGrp'].apply(lambda x: int(42) if x < -20 else 0)
        df['dphiClass']=df['dphiZero'] + df['dphiPos'] + df['dphiNeg'] + df['dphiPosPos'] + df['dphiNegNeg']

        df['drClass']=df['drClass'].astype(int)
        df['dzClass']=df['dzClass'].astype(int)
        df['dphiClass']=df['dphiClass'].astype(int)

        df = df.sort_values(by=['drClass','dzClass','dphiClass'], ascending = True)
        
        return df



#combining 3 class (dr,dz,dphi) into 1 class
def combiningClass(df):
       
       #number of points in one class region
       dfNum = df.groupby(['drClass','dzClass','dphiClass']).size().reset_index(name='numPoints')
       #distinct class region
       dfNum['class']=dfNum.index
       dfMerge = pd.merge(df,dfNum,how='outer',on=['drClass','dzClass','dphiClass'])

       return dfMerge

def describeStatistics(df):
       
       meanI=df.groupby('class')['i'].mean().reset_index(name='i_mean')
       meanJ=df.groupby('class')['j'].mean().reset_index(name='j_mean')
       meanM=df.groupby('class')['m'].mean().reset_index(name='m_mean')
       meanR=df.groupby('class')['r'].mean().reset_index(name='r_mean')
       meanZ=df.groupby('class')['z'].mean().reset_index(name='z_mean')
       meanPhi=df.groupby('class')['phi'].mean().reset_index(name='phi_mean')

       dfs = [df, meanI, meanJ, meanM, meanR, meanZ, meanPhi]
       dfStat = reduce(lambda left,right: pd.merge(left, right, on='class'), dfs)


       return dfStat

#convert to cartesian coordinate to calculate distance between points
def toCartesian(df):

       #x=r*cos(phi radian)
       df.loc[:,'x_cartesian']=df.r.values * np.cos(df.phi.values)

       #y=r*sin(phi radian)
       df.loc[:,'y_cartesian']=df.r.values * np.sin(df.phi.values)

       #z=z
       return df



#create dataframe panel of subset dataframes
def createDict(df,index,range_i,range_j,range_m):

      arrPanel={}
      iRange = int(range_i)
      jRange = int(range_j)
      mRange = int(range_m)
      multi_Index = pd.MultiIndex(levels=[[],[],[]],labels=[[],[],[]],names=[u'i', u'j', u'm'])
      columns_Name = [u'x_cartesian', u'y_cartesian',u'z',u'point']

      if index == 'i':
         num_i=0
         j=0
         m=0
         #in this case mRange=18
         for m in range(0,mRange):
             #in this case jRange=16
             for j in range(0,jRange):
                 i=0
                 panelName='subset_i' + str(num_i)
                 dfsub=pd.DataFrame(index=multi_Index, columns=columns_Name)
                 ##in this case iRange=17
                 for i in range(0,iRange):
                     dfsub=dfsub.append(df.loc[(i,j,m),:]) 
                     i=i+1
                 arrPanel[panelName] = dfsub
                 num_i=num_i+1
                 j=j+1
             m=m+1

      elif index == 'j':
           num_j=0
           i=0
           m=0
           for i in range(0,iRange):
               for m in range(0,mRange):
                   j=0
                   panelName='subset_j' + str(num_j)
                   dfsub=pd.DataFrame(index=multi_Index, columns=columns_Name)
                   for j in range(0,jRange):
                       dfsub=dfsub.append(df.loc[(i,j,m),:]) 
                       j=j+1
                   arrPanel[panelName] = dfsub
                   num_j=num_j+1
                   m=m+1
               i=i+1

      elif index == 'm':
           num_m=0
           i=0
           j=0
           for j in range(0,jRange):
               for i in range(0,iRange):
                   m=0
                   panelName='subset_m' + str(num_m)
                   dfsub=pd.DataFrame(index=multi_Index, columns=columns_Name)
                   for m in range(0,mRange):
                       dfsub=dfsub.append(df.loc[(i,j,m),:]) 
                       m=m+1
                   arrPanel[panelName] = dfsub
                   num_m=num_m+1
                   i=i+1
               j=j+1

      return arrPanel

#calculate distance (delta) r, z, phi of z+1 neighbour (9 points in front of targeted data points)
def calcEuclidian(a,b):

      #nearest neighbour: ml, mr, mc, up, lo, ul, ur, ll, lr (9 points)
      dist = distance.euclidean(a,b)
      return dist

def findNeighbour(strLocation, arrLocation):
      
      for i in range(len(arrLocation)):
          if strLocation in arrLocation[i]:
             result = i
      return result

#create new nearest location features by shifting data point on single grid plane 
#ruf - shifting right, upper, front
#llb - shifting left, lower, back
#processing in pairs (mr,ml); (uc,lc); (fmc,bmc)
def basisNeighbours(arrDictionary, sortedKeys, ruf, llb):
    loc1 = str(ruf)
    loc2 = str(llb)
    for ind in sortedKeys:
        #ruf location (right, upper, front)
        #mr position (i+1,j,m) or
        #uc position (i,j+1,m) or
        #fmc position (i,j,m+1)
        arrDictionary[ind][loc1]=arrDictionary[ind]['point'].shift(-1)

        #llb location (left, lower, back)
        #ml position (i-1,j,m) or
        #lc position (i,j-1,m)
        #bmc position (i,j,m-1)
        arrDictionary[ind][loc2]=arrDictionary[ind]['point'].shift(1)
    
    return arrDictionary

#shifting data in diagonal direction
#upper, lower
#in pairs:
# using frame1:
# ur (i+1,j+1,m) and ll (i-1,j-1,m)
# ul (i-1,j+1,m) and lr (i+1,j-1,m) 
# fur (i+1,j+1,m+1) and bll (i+1,j+1,m+1) 
# fll (i+1,j+1,m-1) and bur (i-1,j-1,m+1)  
# using frame2:
# fuc (i,j+1,m+1) and blc (i,j-1,m-1)
# flc (i,j+1,m-1) and buc (i,j-1,m+1)
# blr (i+1,j-1,m-1) and ful (i-1,j+1,m+1) 
# using frame3:
# fmr (i+1,j,m+1) and bml (i-1,j,m-1)
# fml (i+1,j,m-1) and bmr (i-1,j,m+1)
# bul (i+1,j-1,m+1) and flr (i-1,j+1,m-1) 
def diagNeighbours(frameData, position1, position2):
    #upper position
    strPos1 = str(position1)
    #lower position
    strPos2 = str(position2)
    boundaryIndex = []
    if strPos1 == 'ur' and strPos2 == 'll':
       index1 = [0,0,0]
       index2 = [1,1,0]
    elif strPos1 == 'ul' and strPos2 == 'lr':
       index1 = [1,1,0]
       index2 = [2,0,0]
    elif strPos1 == 'fuc' and strPos2 == 'blc':
       index1 = [0,0,0]
       index2 = [0,1,1]
    elif strPos1 == 'flc' and strPos2 == 'buc':
       index1 = [0,1,1]
       index2 = [0,2,0]
    elif strPos1 == 'fmr' and strPos2 == 'bml':
       index1 = [0,0,0]
       index2 = [1,0,1]
    elif strPos1 == 'fml' and strPos2 == 'bmr':
       index1 = [1,0,1]
       index2 = [2,0,0]
    elif strPos1 == 'fur' and strPos2 == 'bll':
       index1 = [0,0,0]
       index2 = [1,1,1]
    elif strPos1 == 'bul' and strPos2 == 'flr':
       index1 = [0,1,0]
       index2 = [1,0,1]
    elif strPos1 == 'fll' and strPos2 == 'bur':
       index1 = [0,0,1]
       index2 = [1,1,0]
    elif strPos1 == 'blr' and strPos2 == 'ful':
       index1 = [0,1,1]
       index2 = [1,0,0]

    ind1 = frameData[(frameData['i']==index1[0]) & (frameData['j']==index1[1]) & (frameData['m']==index1[2])].index
    ind2 = frameData[(frameData['i']==index2[0]) & (frameData['j']==index2[1]) & (frameData['m']==index2[2])].index
    idx_iFrame = abs(ind1[0] - ind2[0])
    frameData[strPos1]=frameData['point'].shift(-1*idx_iFrame)
    frameData[strPos2]=frameData['point'].shift(idx_iFrame)

    return frameData
 


#natural sorting
import re
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)] 



#main function for data pre-processing
#read ROOT file in python pandas dataframe
#this create matrix table, with index in rows and attributes in columns
#df = rpd.read_root('/home/tita/PyRoot/data/invcorrectness-' +str(sys.argv[1])+ '-' +str(sys.argv[2])+ '-' +str(sys.argv[3])+ '.root', columns=['i','j','m','phi', 'r','z','mCharge','dr','dphir','dz'])
#df = rpd.read_root('/home/tita/PyRoot/data/invcorrectness-17-17-18.root', columns=['phi', 'r','z','mCharge','dr','dphir','dz'])

def extractData(filePath, str_i, str_j, str_m):

   fileName = filePath + 'invcorrectness-' + str_i + '-' + str_j + '-' + str_m + '.root'
   df = rpd.read_root(fileName, columns=['i','j','m','phi', 'r','z','mCharge','dr','dphir','dz'])

#mapping data  
   dfMap = mappingClasses(df)

#combine class
   dfClass = combiningClass(dfMap)

#view statistics
   dfStat = describeStatistics(dfClass)

#map to cartesian coordinate
   dfCartesian = toCartesian(dfStat)

#sorting index
   dfSorted = dfCartesian.sort_values(by=['m','j','i'],ascending=True,inplace=False).reset_index()

   dfLoc=dfSorted[['i','j','m','x_cartesian','y_cartesian','z']] 

#create hierarchical indexing
   dfLoc=dfLoc.set_index(['i','j','m'])

#position in cartesian coordinate - to calculate distance between points
   dfLoc['point']=dfLoc[['x_cartesian','y_cartesian','z']].values.tolist()


#creating new features (nearest neighbours and distances between nearest points) by shifting data
#i.e. for 17*16*18

#nearest neighbours : total 26 
#position of 8-nearest neighours (on the same grid plane):
#ul (upper-left), uc (upper-central), ur (upper-right) 
#ml (mid-left), TARGET POINT {mc (mid-central)}, mr (mid-right)
#ll (lower-left), lc (lower-central), lr (lower-right)

#position of front 9-nearest neighours (m+1):
#ful (front-upper-left), fuc (front-upper-central), fur (front-upper-right) 
#fml (front-mid-left), fmc (front-mid-central), fmr (front-mid-right)
#fll (front-lower-left), flc (front-lower-central), flr (front-lower-right)

#position of back 9-nearest neighours (m-1):
#bul (back-upper-left), buc (back-upper-central), bur (back-upper-right) 
#bml (back-mid-left), bmc (back-mid-central), bmr (back-mid-right)
#bll (back-lower-left), blc (back-lower-central), blr (back-lower-right)

#table type: python dictionary
#creating 3 frames of dictionary to shifting data (to store nearest point location and calculate distance between point and its nearest neighbour)

   dictLoc1 = createDict(dfLoc,'i',str_i,str_j,str_m)
   dictLoc2 = createDict(dfLoc,'j',str_i,str_j,str_m)
   dictLoc3 = createDict(dfLoc,'m',str_i,str_j,str_m)

   dictNames=[dictLoc1,dictLoc2,dictLoc3]
   dictKeys=[]
   sortedKeys=[]
   for i in range(0,3):
       dictKey=dictNames[i].keys()
       sortedKey=sorted(dictKey,key=natural_key)
       dictKeys.append(dictKey)
       sortedKeys.append(sortedKey)

#running 3 engines
#shifting left & right
   dict1 = basisNeighbours(dictNames[0], sortedKeys[0], 'mr', 'ml')
#shifting upper & lower
   dict2 = basisNeighbours(dictNames[1], sortedKeys[1], 'uc', 'lc')
#shifting front & back
   dict3 = basisNeighbours(dictNames[2], sortedKeys[2], 'fmc', 'bmc')


#change back dictionary into dataframe
   subFrame1=[]
   subFrame2=[]
   subFrame3=[]

   subFrames=[subFrame1,subFrame2,subFrame3]
   dicts=[dict1,dict2,dict3]

   for i in range (0,3):
       for ind in range(len(sortedKeys[i])):
           sortedDictKey=sortedKeys[i]
           dictFrame = dicts[i]
           subFrames[i].append(pd.DataFrame(dictFrame[sortedDictKey[ind]]))

   Frame1=pd.DataFrame().append(subFrames[0])
   Frame2=pd.DataFrame().append(subFrames[1])
   Frame3=pd.DataFrame().append(subFrames[2])

   tmpFrame1 = Frame1.reset_index()
   tmpFrame1.mr.fillna(tmpFrame1.point, inplace=True)
   tmpFrame1.ml.fillna(tmpFrame1.point, inplace=True)

   tmpFrame2 = Frame2.reset_index()
   tmpFrame2.uc.fillna(tmpFrame2.point, inplace=True)
   tmpFrame2.lc.fillna(tmpFrame2.point, inplace=True)

   tmpFrame3 = Frame3.reset_index()
   tmpFrame3.fmc.fillna(tmpFrame3.point, inplace=True)
   tmpFrame3.bmc.fillna(tmpFrame3.point, inplace=True)



# using frame1:
# ur (i+1,j+1,m) and ll (i-1,j-1,m)
# ul (i-1,j+1,m) and lr (i+1,j-1,m) 
# fur (i+1,j+1,m+1) and bll (i+1,j+1,m+1) 
# fll (i+1,j+1,m-1) and bur (i-1,j-1,m+1) 

   tmpFrame1 = diagNeighbours(tmpFrame1, 'ur', 'll')
   tmpFrame1.loc[tmpFrame1.i == 16, ['ur']] = np.nan
   tmpFrame1.loc[tmpFrame1.j == 15, ['ur']] = np.nan
   tmpFrame1.ur.fillna(tmpFrame1.point, inplace=True)
   tmpFrame1.loc[tmpFrame1.i == 0, ['ll']] = np.nan
   tmpFrame1.loc[tmpFrame1.j == 0, ['ll']] = np.nan
   tmpFrame1.ll.fillna(tmpFrame1.point, inplace=True)

   tmpFrame1 = diagNeighbours(tmpFrame1, 'ul', 'lr')
   tmpFrame1.loc[tmpFrame1.i == 0, ['ul']] = np.nan
   tmpFrame1.loc[tmpFrame1.j == 15, ['ul']] = np.nan
   tmpFrame1.ul.fillna(tmpFrame1.point, inplace=True)
   tmpFrame1.loc[tmpFrame1.i == 16, ['lr']] = np.nan
   tmpFrame1.loc[tmpFrame1.j == 0, ['lr']] = np.nan
   tmpFrame1.lr.fillna(tmpFrame1.point, inplace=True)

   tmpFrame1 = diagNeighbours(tmpFrame1, 'fur', 'bll')
   tmpFrame1.loc[tmpFrame1.i == 16, ['fur']] = np.nan
   tmpFrame1.loc[tmpFrame1.j == 15, ['fur']] = np.nan
   tmpFrame1.loc[tmpFrame1.m == 17, ['fur']] = np.nan
   tmpFrame1.fur.fillna(tmpFrame1.point, inplace=True)
   tmpFrame1.loc[tmpFrame1.i == 0, ['bll']] = np.nan
   tmpFrame1.loc[tmpFrame1.j == 0, ['bll']] = np.nan
   tmpFrame1.loc[tmpFrame1.m == 0, ['bll']] = np.nan
   tmpFrame1.bll.fillna(tmpFrame1.point, inplace=True)

   tmpFrame1 = diagNeighbours(tmpFrame1, 'fll', 'bur')
   tmpFrame1.loc[tmpFrame1.i == 0, ['fll']] = np.nan
   tmpFrame1.loc[tmpFrame1.j == 0, ['fll']] = np.nan
   tmpFrame1.loc[tmpFrame1.m == 17, ['fll']] = np.nan
   tmpFrame1.fll.fillna(tmpFrame1.point, inplace=True)
   tmpFrame1.loc[tmpFrame1.i == 16, ['bur']] = np.nan
   tmpFrame1.loc[tmpFrame1.j == 15, ['bur']] = np.nan
   tmpFrame1.loc[tmpFrame1.m == 0, ['bur']] = np.nan
   tmpFrame1.bur.fillna(tmpFrame1.point, inplace=True)


# using frame2:
# fuc (i,j+1,m+1) and blc (i,j-1,m-1)
# flc (i,j+1,m-1) and buc (i,j-1,m+1)
# blr (i+1,j-1,m-1) and ful (i-1,j+1,m+1) 

   tmpFrame2 = diagNeighbours(tmpFrame2, 'fuc', 'blc')
   tmpFrame2.loc[tmpFrame2.j == 15, ['fuc']] = np.nan
   tmpFrame2.loc[tmpFrame2.m == 17, ['fuc']] = np.nan
   tmpFrame2.fuc.fillna(tmpFrame2.point, inplace=True)
   tmpFrame2.loc[tmpFrame2.j == 0, ['blc']] = np.nan
   tmpFrame2.loc[tmpFrame2.m == 0, ['blc']] = np.nan
   tmpFrame2.blc.fillna(tmpFrame2.point, inplace=True)

   tmpFrame2 = diagNeighbours(tmpFrame2, 'flc', 'buc')
   tmpFrame2.loc[tmpFrame2.j == 0, ['flc']] = np.nan
   tmpFrame2.loc[tmpFrame2.m == 17, ['flc']] = np.nan
   tmpFrame2.flc.fillna(tmpFrame2.point, inplace=True)
   tmpFrame2.loc[tmpFrame2.j == 15, ['buc']] = np.nan
   tmpFrame2.loc[tmpFrame2.m == 0, ['buc']] = np.nan
   tmpFrame2.buc.fillna(tmpFrame2.point, inplace=True)

   tmpFrame2 = diagNeighbours(tmpFrame2, 'blr', 'ful')
   tmpFrame2.loc[tmpFrame2.i == 0, ['ful']] = np.nan
   tmpFrame2.loc[tmpFrame2.j == 15, ['ful']] = np.nan
   tmpFrame2.loc[tmpFrame2.m == 17, ['ful']] = np.nan
   tmpFrame2.ful.fillna(tmpFrame2.point, inplace=True)
   tmpFrame2.loc[tmpFrame2.i == 16, ['blr']] = np.nan
   tmpFrame2.loc[tmpFrame2.j == 0, ['blr']] = np.nan
   tmpFrame2.loc[tmpFrame2.m == 0, ['blr']] = np.nan
   tmpFrame2.blr.fillna(tmpFrame2.point, inplace=True)


# using frame3:
# fmr (i+1,j,m+1) and bml (i-1,j,m-1)
# fml (i+1,j,m-1) and bmr (i-1,j,m+1)
# bul (i+1,j-1,m+1) and flr (i-1,j+1,m-1) 

   tmpFrame3 = diagNeighbours(tmpFrame3, 'fmr', 'bml')
   tmpFrame3.loc[tmpFrame3.i == 16, ['fmr']] = np.nan
   tmpFrame3.loc[tmpFrame3.m == 17, ['fmr']] = np.nan
   tmpFrame3.fmr.fillna(tmpFrame3.point, inplace=True)
   tmpFrame3.loc[tmpFrame3.i == 0, ['bml']] = np.nan
   tmpFrame3.loc[tmpFrame3.m == 0, ['bml']] = np.nan
   tmpFrame3.bml.fillna(tmpFrame3.point, inplace=True)

   tmpFrame3 = diagNeighbours(tmpFrame3, 'fml', 'bmr')
   tmpFrame3.loc[tmpFrame3.i == 0, ['fml']] = np.nan
   tmpFrame3.loc[tmpFrame3.m == 17, ['fml']] = np.nan
   tmpFrame3.fml.fillna(tmpFrame3.point, inplace=True)
   tmpFrame3.loc[tmpFrame3.i == 16, ['bmr']] = np.nan
   tmpFrame3.loc[tmpFrame3.m == 0, ['bmr']] = np.nan
   tmpFrame3.bmr.fillna(tmpFrame3.point, inplace=True)

   tmpFrame3 = diagNeighbours(tmpFrame3, 'bul', 'flr')
   tmpFrame3.loc[tmpFrame3.i == 0, ['bul']] = np.nan
   tmpFrame3.loc[tmpFrame3.j == 15, ['bul']] = np.nan
   tmpFrame3.loc[tmpFrame3.m == 0, ['bul']] = np.nan
   tmpFrame3.bul.fillna(tmpFrame3.point, inplace=True)
   tmpFrame3.loc[tmpFrame3.i == 16, ['flr']] = np.nan
   tmpFrame3.loc[tmpFrame3.j == 0, ['flr']] = np.nan
   tmpFrame3.loc[tmpFrame3.m == 17, ['flr']] = np.nan
   tmpFrame3.flr.fillna(tmpFrame3.point, inplace=True)

   tmp_Frame2=tmpFrame2[['i','j','m','uc','lc','fuc','blc','buc','flc','blr','ful']]
   tmp_Frame3=tmpFrame3[['i','j','m','fmc','bmc','fmr','bml','bmr','fml','flr','bul']]
   frames = [tmpFrame1,tmp_Frame2,tmp_Frame3]
   allFrame = reduce(lambda left,right: pd.merge(left, right, on=['i','j','m']), frames)

   arrNeighbours=['mr','ml','ur','lr','ll','ul','uc','lc', 'fmr','fml', 'fmc', 'fur','flr','fll','ful','fuc','flc', 'bmr','bml', 'bmc', 'bur','blr','bll','bul','buc','blc']

   for i in range(0,len(arrNeighbours)):
       strNeighbour = str(arrNeighbours[i])
       strDist = 'd_' + strNeighbour
       allFrame[strDist] = allFrame.apply(lambda row: distance.euclidean(row['point'],row[strNeighbour]), axis=1)

   #merge dataframe with euclidian distance and original dataframe
   df1=dfSorted[['i', 'j', 'm', 'phi', 'r', 'z', 'mCharge', 'dr','dphir','dz','drClass','dphiClass','dzClass','class']]

   df2=allFrame[['i','j','m','x_cartesian', 'y_cartesian','d_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc']]

   dfDat = pd.merge(df1,df2,how='outer',on=['i','j','m'])


#multi-dimensional continuous classes
   dat1=dfDat[['i', 'j', 'm', 'x_cartesian', 'y_cartesian', 'z','mCharge', 'd_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc', 'dr','dphir','dz']]
   dat2=dfDat[['phi', 'r', 'z', 'mCharge', 'd_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc', 'dr','dphir','dz']]

#discretized class
   dat3=dfDat[['i', 'j', 'm', 'x_cartesian', 'y_cartesian', 'z', 'mCharge', 'd_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc', 'class']]
   dat4=dfDat[['phi', 'r', 'z', 'mCharge','d_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc', 'class']]


#multidimensional discretized class
   dat5=dfDat[['i', 'j', 'm', 'x_cartesian', 'y_cartesian', 'z', 'mCharge', 'd_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc', 'drClass', 'dzClass', 'dphiClass']]
   dat6=dfDat[['phi', 'r', 'z', 'mCharge','d_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc', 'drClass', 'dzClass', 'dphiClass']]

   return dat1



