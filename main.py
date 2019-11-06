# Density Based Spatial Clustering of Applications with Noise

import numpy as np 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
import sys


NO_args=len(sys.argv)
print ('Number of arguments:', NO_args, 'arguments.')
print ('Argument List:', str(sys.argv))
if NO_args!=6:
    exit('number of args !!')


try:
    show= str(sys.argv[1])
    save= str(sys.argv[2])
    data_path= str(sys.argv[3])
    Epsilon= float(sys.argv[4])
    MinPts= int(sys.argv[5])

except ValueError:
    print("Args !!")

print(show,save,data_path,Epsilon,MinPts)
#load the dataset
weather_df = pd.read_csv(data_path)
print ("[*] Shape of the DataFrame: ", weather_df.shape)
weather_df.head(3) 

#drop the rows that contain NaN values in the columns: Tm, Tn and Tx.
weather_df.dropna(subset=['Tm', 'Tx', 'Tn'], inplace=True)
print ("[*] After Dropping Rows that contains NaN on Mean, Max, Min Temperature Column: ", weather_df.shape)

#select the boundaries of the map from lattitude and longitude
rcParams['figure.figsize'] = (14,10)
(llon,ulon,llat,ulat)=(-140,-50,40,75)
weather_df = weather_df[(weather_df['Long'] > llon) & (weather_df['Long'] < ulon) & (weather_df['Lat'] > llat) &(weather_df['Lat'] < ulat)]

#create the map
my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
my_map.drawlsmask(land_color='orange', ocean_color='skyblue')
my_map.bluemarble()

# To collect data based on stations        
xs,ys = my_map(np.asarray(weather_df.Long), np.asarray(weather_df.Lat))
weather_df['xm']= xs.tolist()
weather_df['ym'] =ys.tolist()

#Visualization1
for index,row in weather_df.iterrows():
   my_map.plot(row.xm, row.ym,markerfacecolor ='lime',markeredgecolor='pink', marker='s', markersize= 10, alpha = 0.4)
plt.title("Weather Stations in Canada", fontsize=14)
if save:
    plt.savefig("Canada_WS.png", dpi=300)
if show:
    plt.show()

#stats
print ("[*] stats:")
print ("[*] Minimum  temp: ", weather_df['Tm'].min())
print ("[*] Minimum  Temp: ", weather_df['Tx'].min())
print ("[*] Maximun  Temp: ", weather_df['Tn'].max())

sns.distplot(weather_df['Tm'], color='purple', kde=False)
plt.xlabel('Mean Temperature (Â°C)', fontsize=14)
plt.title("Distribution of Mean Temperature", fontsize=14)
if save:
    plt.savefig("Dist_of_Mean_Temp.png", dpi=200)
if show:
    plt.show()


#Proceed To Clustering using DBSCAN
weather_df_clus_temp = weather_df[["Tm", "Tx", "Tn", "xm", "ym"]]
weather_df_clus_temp = StandardScaler().fit_transform(weather_df_clus_temp)
db = DBSCAN(eps=Epsilon, min_samples=MinPts).fit(weather_df_clus_temp)
labels = db.labels_
print ("[*] number of labels:   ",labels)
#printing labels
weather_df["Clus_Db"]=labels
realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))
print("realClusterNumber",realClusterNum)
print("ClusterNumber",clusterNum)
print("set(labels)",set(labels))
set(labels)
rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
my_map.drawlsmask(land_color='orange', ocean_color='skyblue')
my_map.etopo()

# To create a color map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

#Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = weather_df[weather_df.Clus_Db == clust_number]                    
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 40, alpha = 0.65)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=30, color='red',)
        print ("Cluster "+str(clust_number)+', Average Mean Temp: '+ str(np.mean(clust_set.Tm)))
plt.title(r"Weather Stations in Canada Clustered (1): $ \epsilon = ",Epsilon", fontsize=14)        
if save:
    plt.savefig("etopo_cluster.png", dpi=300)        
if show:
    plt.show()