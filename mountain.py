#!/usr/local/bin/python3
#
# Authors: [SANATH KEERTHI EDUPUGANTI(saedup),SAI PRASAD PARSA(saiparsa),AKHIL NAGULAVANCHA(aknagu)]
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2019
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import numpy as np
# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = np.array(input_image.convert('L'))
    filtered_y = np.zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return np.sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

# main program
#
(input_filename, r, c) = sys.argv[1:]
r = int(r)
c = int(c)
# load in image 
input_image = Image.open(input_filename)
input_image1= Image.open(input_filename)
input_image2= Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imageio.imwrite('edges.jpg', np.uint8(255 * edge_strength / (np.amax(edge_strength))))


edges=edge_strength
edges1=np.array(edges)
k=np.zeros(len(edges1[0]))
# ridge using gradient matrix
def ridge(c):
  k=np.zeros(len(c[0]))
  for i in range(len(c[0])):
    p=c[:,i]
    result = np.where(p == np.amax(p))[0]
    r=result.tolist()
    k[i]=r[0]
  return k
#optput for max of edge matrix  
ridge1 = ridge(edges1)
imageio.imwrite("output_edge.jpg", np.array(draw_edge(input_image,ridge1, (255, 0, 0), 5)))
#ridge using viterbi algorithm
J=np.zeros(len(edges1))
for i in range(len(J)):
  J[i]=i
#transition matrix generation
def matrix(x,y):
  G=1/(1+((y-x)**4))
  G=G/sum(G)
  a=0.9999
  
  if x==0:
    for i in range(x,x+7):
      G[i]=(a/7)
    
    G[x+7:]=((1-a)/(len(G)-7))
  if x==1:
    for i in range(x-1,x+6):
      G[i]=(a/7)
    #s=sum(G[x-1:x+7])  
    G[:x-1]=((1-a)/(len(G)-7))
    G[x+6:]=((1-a)/(len(G)-7))
  if x==2:
    for i in range(x-2,x+5):
      G[i]=(a/7)
    #s=sum(G[x-1:x+6])  
    G[:x-2]=((1-a)/(len(G)-7))
    G[x+5:]=((1-a)/(len(G)-7))
  if x>3 and x<(len(G)-4):
    for i in range(x-3,x+4):
      G[i]=(a/7)
    #s=sum(G[x-3:x+4])  
    G[:x-3]=((1-a)/(len(G)-7))
    G[x+4:]=((1-a)/(len(G)-7))
  if x==(len(G)-4):
    for i in range(x-3,len(G)):
      G[i]=(a/7)
    G[:x-3]=((1-a)/(len(G)-7))
  if x==(len(G)-3):
    for i in range(x-4,len(G)):
      G[i]=(a/7)
    G[:x-4]=((1-a)/(len(G)-7))
  if x==(len(G)-2):
    for i in range(x-5,len(G)):
      G[i]=(a/7)
    G[:x-5]=((1-a)/(len(G)-7))
  if x==(len(G)-1):
    for i in range(x-6,len(G)):
      G[i]=(a/7)
    G[:x-6]=((1-a)/(len(G)-7))

  return G
transition_mat=[]
for i in range(len(edges1)):
  T=matrix(i,J)
  transition_mat.append(T)
transition_mat=np.asarray(transition_mat)
emission=[]
for i in range(len(edges1[0])):
  emission.append(edges1[:,i]/sum(edges1[:,i]))
parent=[]
emission1=np.asarray(emission)
def probgen(r):
  V_mat=[]
  for i in range(len(edges1)):
    v=(r[i]) + np.log(transition_mat[i])
    V_mat.append(v)
  V_mat=np.asarray(V_mat)
  V_mat_f=np.zeros(len(edges1))
  for i in range(len(edges1)):
    V_mat_f[i]=np.amax((V_mat[:,i]))
  parent.append(np.argmax((V_mat[:,i])))
  return V_mat_f,parent
O=[]
O.append(np.log(emission[0]))
P=[]
for i in range(1,len(edges1[0])):
  VT,p=probgen(O[i-1])
  o= VT + np.log(emission[i])
  O.append(o)
  P.append(p)
O=np.asarray(O)
O=np.transpose(O)
ridge2=ridge(O)
# output for viterbi algorithm
imageio.imwrite("output_map.jpg", np.array(draw_edge(input_image1, ridge2, (0,255, 0), 5)))
#for human input
def humaninput(image,x,y):
  #emission1=
  emission1[y][x-1:x+2]=1
  emission1[y][:x-1]=0
  emission1[y][x+2:]=0
  output_human=[]
  output_human.append(np.log(emission1[0]))
  for i in range(1,len(edges1[0])):
    VT,p=probgen(output_human[i-1])
    out= VT + np.log(emission1[i])
    output_human.append(out)
    P.append(p)
  output_human=np.asarray(output_human)
  output_human=np.transpose(output_human)
  ridge3=ridge(output_human)
  imageio.imwrite("output_human.jpg", np.array(draw_edge(input_image2,ridge3, (0,0,255), 5)))
#for human input
humaninput(input_image2,r,c)
