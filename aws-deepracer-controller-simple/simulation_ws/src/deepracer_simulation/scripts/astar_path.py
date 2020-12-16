# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

##############################################################################

# import packages

##############################################################################

 

import numpy as np

import heapq

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

from scipy.signal import savgol_filter

import pandas as pd
import os
import json
 

##############################################################################

# plot grid

##############################################################################

 

grid = np.array([

    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],

    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# start point and goal
#grid = np.pad(grid,(1,1),'constant',constant_values=1)
grid = np.zeros((80,80))
grid = np.pad(grid,(1,1),'constant',constant_values=1)
#grid[11:94,3]=1
#grid[0:50,30]=1
#grid[30:100,60]=1
'''
grid[50,:]=1
grid[:,50]=1
grid[50,35:49]=0
grid[15:30,50]=0
grid[50,75:90]=0
grid[75:90,50]=0
grid[10:15,10:15]=1
grid[10:15,40:45]=1
grid[90:92,10:13]=1
grid[80:95,65:85]=1
grid[10:15,10:15]=1
grid[70:80,35:40]=1
grid[30:40,70:95]=1
grid[30:35,10:25]=1
'''
grid[1:25,41]=1
grid[37:69,41]=1
grid[40,1:17]=1
grid[40,29:41]=1
grid[40,53:81]=1
grid[17:25,17:25]=1
grid[9:17,53:69]=1
grid[57:65,21:33]=1
grid[53:61,49:57]=1

start = (5,5)

goal = (75,65)


##############################################################################

# heuristic function for path scoring

##############################################################################

 

def heuristic(a, b):
 
    
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    #return sum((abs(b[0]-a[0]),abs(b[1]-a[1])))
 
def n_score(grid,a):
    score = 0
   
    '''
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    for n in neighbors:
         if a[0]+n[0]>=0 and a[0]+n[0]<grid.shape[0] and a[1]+n[1]>=0 and a[1]+n[1]<grid.shape[1]:
             score_1 = score_1+grid[a[0]+n[0],a[1]+n[1]]
    '''     
    for i in range(-6,6):
        for j in range(-6,6):
            if i!=0 and j!=0:
                if a[0]+i>=0 and a[0]+i<grid.shape[0] and a[1]+j>=0 and a[1]+j<grid.shape[1]:
                    score = score+(10/abs(i)+10/abs(j))*grid[a[0]+i,a[1]+j]
    return score
##############################################################################

# path finding function

##############################################################################

 

def astar(array, start, goal):

    #neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    #neighbors = [(0,1),(0,-1),(1,0),(-1,0)]

    close_set = set()

    came_from = {}

    gscore = {start:n_score(grid,start)}
    #gscore_1 = {start:0}

    fscore = {start:heuristic(start, goal)}

    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
 

    while oheap:

        current = heapq.heappop(oheap)[1]

        if current == goal:

            data = []

            while current in came_from:

                data.append(current)

                current = came_from[current]

            return data

        close_set.add(current)

        for i, j in neighbors:

            neighbor = current[0] + i, current[1] + j
            #tentative_g_score = gscore[current] + heuristic(current, neighbor)
            tentative_g_score_1 = gscore[current] + heuristic(current, neighbor) 
            tentative_g_score =   tentative_g_score_1 + n_score(grid, neighbor)
            #tentative_g_score = gscore[current] + heuristic(current, neighbor)
            #print(neighbor, gscore_1[current],gscore[current],n_score(grid, neighbor))
            
            if 0 <= neighbor[0] < array.shape[0]:

                if 0 <= neighbor[1] < array.shape[1]:                

                    if array[neighbor[0]][neighbor[1]] == 1:

                        continue

                else:

                    # array bound y walls

                    continue

            else:

                # array bound x walls

                continue
 

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):

                continue
 

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:

                came_from[neighbor] = current

                #gscore_1[neighbor] = tentative_g_score_1
                gscore[neighbor] = tentative_g_score_1
                

                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                heapq.heappush(oheap, (fscore[neighbor], neighbor))
 

    return False

route = astar(grid, start, goal)

route = route + [start]

route = route[::-1]
#smoothroute = []
'''
for i in range(len(route)):
     neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
     plist=[]
     score = 0
     for n in neighbors:
         if grid(route[i][0]+n[0],route[i][1]+n[1])==0
             plist.append((route[i][0]+n[0],route[i][1]+n[1]))
         score = score+grid(route[i][0]+n[0],route[i][1]+n[1])
     for it in plist:
         tmp = 0 
         r = route[i]
         for n in neighbors:
             tmp = tmp+grid(it[0]+n[0],it[1]+n[1])
         if i-1>=0 and i+1<len(route):
             if tmp<score and it!=route[i-1] and it!=route[i+1]: 
                r=it
                score = tmp
                astar(grid,r,route[i+1] 
             smroute.append(r
'''
#print(route)


def get_obstacle_coordinates(graph):
    #print("here")
    #print(graph)
    adj_matrix = np.asmatrix(graph)
    adj_matrix = np.delete(graph,(0),axis=0)
    adj_matrix = np.delete(adj_matrix,(-1),axis=0)
    adj_matrix = np.delete(adj_matrix,(-1),axis=1)
    adj_matrix = np.delete(adj_matrix,(0),axis=1)
    print(pow(len(adj_matrix),2))
    obstacles = []
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] == 1:
                #print(i,j)
                if(i==0 or j == 0):
                    obstacles.append([i+1,j+1])
                elif(adj_matrix[i+1][j]==0 or adj_matrix[i-1][j]==0 or adj_matrix[i][j+1]==0 or adj_matrix[i][j-1] == 0):
                    obstacles.append([i+1,j+1])

    print(len(obstacles))
   
    for i in range(len(obstacles)):
        plt.plot(obstacles[i][1],obstacles[i][0],'.')
        plt.title("Obstacle boundary used for adding obstacle constraint")
        #plt.hold(True)
    plt.show()
    
    #print(obstacles[:,0])
    if os.path.isfile("obstacles_a_star.txt"):
        os.remove("obstacles_a_star.txt")
    #np.savetxt("obstacles_a_star.txt",(obstacles[:,0],obstacles[:,1]),fmt=%d)
    with open('obstacles_a_star.txt','w') as f:
        for item in obstacles:
            f.write(str(item[0])+" "+str(item[1])+"\n")
    
    #x = np.loadtxt("obstacles_a_star.txt")

                
##############################################################################

# plot the path

##############################################################################

 

#extract x and y coordinates from route list

x_coords = []

y_coords = []

for i in (range(0,len(route))):

    x = route[i][0]

    y = route[i][1]

    x_coords.append(x)

    y_coords.append(y)

# plot map and path

fig, ax = plt.subplots(figsize=(20,20))

ax.imshow(grid, cmap=plt.cm.Dark2)

ax.scatter(start[1],start[0], marker = "*", color = "yellow", s = 200)

ax.scatter(goal[1],goal[0], marker = "*", color = "red", s = 200)


x_f = savgol_filter(x_coords, 11, 3)
y_f = savgol_filter(y_coords, 11, 3)

df = pd.DataFrame({"X" : x_f, "Y" : y_f})
df.to_csv("route.csv", index=False)

get_obstacle_coordinates(grid)

ax.plot(y_f,x_f, color = "red")
#ax.plot(y_coords,x_coords, color = "black")

plt.show()


