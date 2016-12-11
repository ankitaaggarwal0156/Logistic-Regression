'''
Created on Oct 18, 2016

@author: Ankita
'''
import numpy  as np
from numpy import random
Data_list = []
output_coordinate_list=[]
alpha=0.001
w =random.rand(4)
count=0
list_point=[]
weightMatrix=[]
def ReadFile():
    list_of_coordinates = []
    global Data_list
    global output_coordinate_list
    with open("./classification.txt", "r") as fo:
        for line in fo:
            list_of_coordinates.append(line)
    fo.close()
    for line in list_of_coordinates:
        list_of_items_in_line = line.split(",")
        Data_list.append([1,float(list_of_items_in_line[0]),float(list_of_items_in_line[1]),float(list_of_items_in_line[2])])
        output_coordinate_list.append(int(list_of_items_in_line[4]))  

    for i in range(len(output_coordinate_list)):
        if (output_coordinate_list[i]==-1):
            output_coordinate_list[i]=0
    


def sigmoid(z):  
    return 1 / (1 + np.exp(-z))



def CountUnclassified(theta, X, y): 
    theta = np.matrix(theta)
    #print theta
    countClassify=0
    for i in range(len(X)):
        probability = sigmoid(X[i] * theta.T)
        #print "pppppp", probability
        if (probability>0.5 and y[i] ==1):
            countClassify=countClassify+1
        elif (probability<0.5 and y[i] ==0):
            countClassify=countClassify+1       
    return (len(Data_list)-countClassify)        




          
def Logistic():
    global count, m,iter1, list_point, w, Data_list, output_coordinate_list, alpha
    dw=[]
    m=len(Data_list)
    w = np.matrix(w)
    term=0.0
    parameters = int(w.ravel().shape[1])
    Weightupdate(w)
    #print " Weight Matrix =", weightMatrix
    while (count<7000):
        
        #print " while w", w
        count+=1 
        #dw = np.zeros(parameters)
        for t in range (len(Data_list)):
            dw = np.zeros(parameters)
            y = sigmoid(Data_list[t] * w.T)
            #print "hhhereeee", y
            error = y - output_coordinate_list[t]
            #print "error ", error
            term = np.multiply(error, Data_list[t])
            dw = dw+ term
            #print "dw = ", dw
            if (y>0.5 and output_coordinate_list[t] ==0):
                w= w- alpha*dw
            elif (y<0.5 and output_coordinate_list[t] ==1):
                w= w- alpha*dw
            
            
            
        #print "New w", w
            
        n = CountUnclassified(w, Data_list, output_coordinate_list) 
        print " unclassified points", n  
        list_point.append(n) 
        Weightupdate(w)
        if(n<m):
            m = n
            iter1 = count
            
        if(m==0):
            break
    if(count ==7000 or m==0):
        #print iter1
        print "\n------------Output--------------------------------\n"
        print "W final ", weightMatrix[iter1-1]   
        print " Iteration # for minimum Misclassified", iter1
        print "minimum misclassified constraints", m
        
        



def Weightupdate(m):
    global weightMatrix
    temp=m[:]
    weightMatrix.append(temp)

            
ReadFile()

n = CountUnclassified(w, Data_list, output_coordinate_list) 
#print " unclassified points", n  
list_point.append(n)
print "Inital Random Weight:\n", w

Logistic()
