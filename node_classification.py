import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os
import re
import cv2
import random

Nbins=range(50,65,5)
th_strength=10 #threshold of energy: we discard measurements of lower strength
th_depth=4 #threshold of depth (in meters) in the binary sequence
  

nodes=["NoNode","EndNode","NodeT","CrossNode","NodeL", "OpenNode"]                     
               
it=0
def node_classification(file_scan,nbin,flag_signature):
     #capture=captures[i]
     #path_total=os.path.join(path_seq,file_scan)
     
     scan=np.load(file_scan,allow_pickle=True,encoding='latin1')
     filename=file_scan
     
     #Si LIDAR al revés en plataforma)
     '''
     for n in range(len(scan[:,1])):
          if scan[n,1]>180:
               scan[n,1]-=180
          else:
               scan[n,1]+=180
     degrees=scan[:,1]-180 
     '''
     degrees=scan[:,1]-180 
     depth=scan[:,2]
     strength=scan[:,0]
     #Ordeno indices para que en representacion no salgan valores cruzados en plot
     sort_indexes = np.argsort(degrees)
     degrees=np.array ([degrees[n] for n in sort_indexes])  
     depth= np.array([depth[n] for n in sort_indexes])
     strength= np.array([strength[n] for n in sort_indexes])
     #filtered data discarding measurements of low pulse reflected strength
     depth_filtered=depth[np.where(strength>th_strength)]
     degrees_filtered=degrees[np.where(strength>th_strength)]
     #Compute the continous function (interpolation of samples)                          
     #Consider the interval of angles [-130--130] to interpolate the function. 
     #If there are not samples above 90º or below -90º, skip the capture. We cannot make the interpolation in the range[-90º--90º].
     deg_fit=degrees_filtered[(np.where(degrees_filtered>=-130)) and (np.where( degrees_filtered<=130) )]
     depth_fit=depth_filtered[(np.where(degrees_filtered>=-130)) and (np.where( degrees_filtered<=130) )] 
     funct = interp1d(deg_fit,depth_fit, kind='slinear') #kind(other possibilities): 'linear','cubic',... 
     if (np.min(deg_fit)>-90.0 or np.max(deg_fit)<90): 
          print('skip filename')
          plt.close('all')
          return
     #Si flag_signature==True hago la representacion     
     if flag_signature:
          fig = plt.figure(it)
          #Representation of raw data
          fig.add_subplot(411)  
          plt.title(filename)         
          plt.plot(degrees,depth/1000,'r-o',linewidth=1,markersize=3)
          plt.xlabel('Degrees',fontsize=8)
          plt.ylabel('Dist (m)',fontsize=8)
          plt.grid(True)

          plt.yticks(fontsize=8)
          plt.xticks(fontsize=8)
          plt.ylim(0, 14)
          plt.xlim(-90, 90)

          #Representation of filtered data discarding measurements of low strength
          fig.add_subplot(412) 
          plt.plot(degrees_filtered,depth_filtered/1000,'r-o',linewidth=1,markersize=3)
          plt.xlabel('Degrees',fontsize=8)
          plt.ylabel('Dist (m)',fontsize=8)
          plt.grid(True)
          plt.yticks(fontsize=8)
          plt.xticks(fontsize=8)
          plt.ylim(0, 14)
          plt.xlim(-90, 90)


          #Compute the continous function (interpolation of samples)
          fig.add_subplot(413) 
          x=np.arange(-90,90,0.01)
          plt.ylabel('Dist(m)',fontsize=8)
          plt.plot(x,funct(x)/1000,  color='red')
          plt.ylim(0, 14)
          plt.xlim(-90, 90)
          plt.grid(True)
          #draw the threshold of distance
          plt.plot(x,th_depth*np.ones(len(x)),color='b')


          #Reflected pulse strength
          fig.add_subplot(414)           
          plt.plot(degrees,strength,'b-o',linewidth=1,markersize=3)
          plt.xlabel('Degrees',fontsize=8)
          plt.ylabel('Strength',fontsize=8)
          plt.grid(True)
          plt.yticks(fontsize=8)
          plt.xticks(fontsize=8)
          plt.ylim(0, 17)
          plt.xlim(-90,90)
          plt.ylabel('Strength',fontsize=8)
          #path=os.path.join(path_dataset,directory_node)
          path_signature=os.path.join(path_seq,'LidarSignature')
          fileimg=os.path.join(path_signature,filename[:-4]+'.png')
          plt.savefig(fileimg)
                                                  
          plt.close('all')
 
     bins=[]
     bins_analog=[]
     range_interval=180/nbin
     for p in range(nbin):
          i1=p*range_interval-90
          i2=(p+1)*range_interval-90
          #print(i,i1,i2)
          n=0   
          k=0
          for q in np.arange(i1,i2,0.1):
               k+=1
               if funct(q)/1000>th_depth:
                    n+=1
    
          if (n/k)>=0.75:
               bins.append(0) #Occupancy (0: empty, 1: occupied)
          else:
               bins.append(1) 
          bins_analog.append(funct((i1+i2)/2)/1000)     
              
     '''              
     #Detect indexes of transition in the binary sequence
     ind1=[] #indexes of starting of empty area
     ind2=[] #indexes of starting of empty area
              
     for n in range(1,len(bins)):
          if n-1==0 and bins[n-1]==0:
               ind1.append(0)
          if bins[n-1]==1 and bins[n]==0:
               ind1.append(n)
          if bins[n-1]==0 and bins[n]==1:
               ind2.append(n-1)
          #if i==len(bins)-1 and bins[n]==0:
          #     ind2.append(n)
                                         
       '''

               ##############################
               
          ##################################################
          #Representation of image and Lidar signature
          ##################################################
     if flag_signature: 
          fileimgRGB=file_scan.replace('Capture_Lidar','ImgTestResults')
          fileimgRGB=fileimgRGB.replace('Lidar','ImgColor')
          fileimgRGB=fileimgRGB.replace('npy','png')
          #fileimgRGB=fileimgRGB+'.png'
          imgRGB=cv2.imread(fileimgRGB)     
          fig = plt.figure()
          #plt.title(path[-8:],fontsize=14)
          theta=np.arange(-90,90,0.01)
          plt.ylabel('depth (m)',fontsize=18)
          plt.xlabel('angle (degrees)',fontsize=18)
          plt.plot(theta,funct(theta)/1000,  color='b',linewidth=7)
          plt.ylim(0, 14)
          plt.xlim(-90, 90)
          plt.grid(True)
          plt.plot(theta,th_depth*np.ones(len(theta)),color='r',linestyle='dashed',linewidth=3)
          #plt.plot(deg_fit,depth_fit/1000,'k*')
          y=np.arange(0,14,0.1)
          for i in range(nbin):
               i1=i*range_interval-90
               i2=(i+1)*range_interval-90
               plt.plot(i1*np.ones(len(y)),y,color='m',linestyle='dashed')
               x_pos=(i1+i2)/2-3
               y_pos=10
               if bins[i]==0: #inicialmente consideraba '0' como libre y '1' ocupado: en ECMR consideré cambiar criterio: '1' libre para explorar y '0' no libre para explorar
                    val=1
               else:
                    val=0
               plt.text(x_pos, y_pos, str(val),fontsize=16)
     

          plt.savefig("bins_aux.png")
          plt.close('all')
          imgLidar=cv2.imread("bins_aux.png")
          h1,w1,_=imgRGB.shape
          h2,w2,_=imgLidar.shape
          imgLidar2= np.zeros((h1,w2,3), dtype = "uint8")
          imgLidar2[0:h2,0:w2]=imgLidar
          img_res = cv2.hconcat([imgRGB, imgLidar2])
          path_RGBLidar=os.path.join(path_seq,'RGBLidar')
          #path_RGBLidar=os.path.join(path_RGBLidar,str(nbin)+'bins')
          filename_total=os.path.join(path_RGBLidar,filename[:-4]+'_RGBLidar.png')
          cv2.imwrite(filename_total,img_res)
          
     ##########################################
     #Write training/test vectors bins_analog[n]/12
     #for n in range(nbin):
     #     print("n:%d,%0.4g" %(n,bins_analog[n]/12))
     return(bins_analog)                     
 
####################################################
#MAIN PROGRAM
####################################################

