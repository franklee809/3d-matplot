#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

# initialize the graph

def initializeGraph():
    x = data['x']
    y = data['y']
    
    plt.scatter(x, y)
    plt.plot(plot1[0], plot1[1], 'go-', label='line 1', linewidth=2)
    plt.plot(plot2[0], plot2[1], 'go-', label='line 2', linewidth=2)
    
    plt.show()

def initializeGraphZ():
    
    new = data.loc[data['sensor'] == 'crack']
    
    y   = new['x'].drop_duplicates().values 
    z   = [0] * len(y)  

    arr = new['z'].drop_duplicates().values
    y   = np.append(y, y)
    z   = np.append(z, arr)
    
    new_dict = {}
    for i in range(0,len(y)):
      if(y[i] in new_dict):
        new_dict[y[i]].append([y[i], z[i]])
      else: 
        new_dict[y[i]] = [[y[i], z[i]]]
        
    for i in new_dict:
      # print(new_dict[i])
      t = new_dict[i]
      plt.plot([t[0][0], t[1][0]], [t[0][1], t[1][1]], 'go-', linewidth=2)
    
def checkNodesById():
    start = 0
    end = 0
    for i in range(1, 7):
        start = i
        end = i + 3
        checkVelocity(start, end)


def checkVelocity(start, end):
    records = location['velocity'].loc[(location['from'] == start)
            & (location['to'] == end) | (location['to'] == start)
            & (location['from'] == end)].tolist()

    arr = location.loc[(location['from'] == start) & (location['to']
                       == end) | (location['to'] == start)
                       & (location['from'] == end)]

    notMiddleLine = compareVelocity(start, end, arr)

  # draw middle line

    if records[0] == records[1] and records[0] \
        == crack_line_node_balance:
          
        new_node = createNewNode(start, end)
        setGlobalData(new_node)
        initializeGraph()
        
    elif notMiddleLine:
      
        middle = getMiddleLine(start, end)
        side_crack = storeSideCrack(notMiddleLine, middle, start)
        new_node = createNewSideNode(start, side_crack, end)
        setGlobalData(new_node)
        initializeGraph()
        
    else:
        print(start, end)
        print('velocity not in range'+"\n")


def createNewNode(start, end):
    sum_of_sensor = data['x'].loc[(data['sensor'] == start)
                                  | (data['sensor'] == end)].sum()
    velocity      = location['velocity'].loc[(location['from'] == start)
                                  & (location['to'] == end)]
     
    crack_location_x = round((sum_of_sensor / 2).astype(int), 2)
    crack_location_y = round(data['y'].loc[data['sensor']
                             == start].values[0], 2)
    crack_location_z = calculateDepth(velocity.values)
    
    new_index = data.iloc[-1].name + 1

    new_node = [['crack', crack_location_x, crack_location_y, crack_location_z[0]]]
    storeCrackLine(crack_location_x, crack_location_y)  # store line graph

    new_data_frame = pd.DataFrame(new_node, columns=['sensor', 'x', 'y', 'z'
                                  ])
    new_data_frame.name = new_index

    return new_data_frame


def createNewSideNode(start, crack, end):
    velocity      = location['velocity'].loc[(location['from'] == start)
                                  & (location['to'] == end)]

    crack_location_x = round(crack, 2)
    crack_location_y = round(data['y'].loc[data['sensor']
                             == start].values[0], 2)

    crack_location_z = calculateDepth(velocity.values)


    new_index = data.iloc[-1].name + 1
    new_node = [['crack', crack_location_x, crack_location_y, crack_location_z[0]]]
    storeCrackLine(crack_location_x, crack_location_y )  # store line graph

    new_data_frame = pd.DataFrame(new_node, columns=['sensor', 'x', 'y', 
                                                     'z'
                                  ])
    new_data_frame.name = new_index

    return new_data_frame


def storeCrackLine(x, y):
    plot1[0].append(x)
    plot1[1].append(y)


def setGlobalData(new_data):
    global data
    new = data.append(new_data)
    data = new


def compareVelocity(start, end, array):
    from_to = array['velocity'].loc[(location['from'] == start)
                                    & (location['to'] == end)].iloc[0] \
                                    - 1328
    to_from = array['velocity'].loc[(location['to'] == start)
                                    & (location['from']
                                    == end)].iloc[0] - 1328

    
    if(from_to  >= max_velocity or from_to <= min_velocity):
      return False
    
    if(to_from >= max_velocity or to_from <= min_velocity):
      return False
    
    if abs(from_to) == abs(to_from):
      return from_to
    else:
      return False


def storeSideCrack(velocity, middle, start):
    global max_velocity, min_velocity, radius_bet_two_sensor
    node = 0

  # check positive and smaller than max velocity

    if velocity < max_velocity and velocity >= 1:
        node = velocity / max_velocity * radius_bet_two_sensor
    elif velocity > min_velocity and velocity <= -1:

  # check negative and smaller than min velocity

        node = -(velocity / min_velocity) * radius_bet_two_sensor

    return middle + node


def getMiddleLine(start, end):
    global data
    incre = 10
    arr = data['x'].loc[(data['sensor'] == start) | (data['sensor']
                        == end)]

    if arr.iloc[1] > 15:
        return arr.iloc[1] - arr.iloc[0] + incre

    return arr.iloc[1] - arr.iloc[0]


# calculate Z
def calculateDepth(velocity):
  depthOfCrack = (velocity/ (frequency * 1000 * 2) ) * 1000
  print('depth of crack= '+ str(depthOfCrack[0]))
  return depthOfCrack

def threeDimensionGraph():
  global data, plot
  fig = plt.figure()
  ax  = plt.axes(projection='3d')
    
  # scatter 
  xdata = pd.concat([data['x'], pd.Series(plot1[0])], axis = 0)     # left bottom  hand
  ydata = pd.concat([data['y'], pd.Series(plot1[1])], axis = 0)     # right bottom hand
  zdata = pd.concat([data['z'], pd.Series([0,0,0])], axis = 0)      # right side
  
  # line graph
  for i in range(2):
    xline = plot1[0]
    yline = plot1[1]
    if(i == 0):
      zline = [0,0,0]
    else:
      zline = data['z'].loc[data['sensor'] == 'crack']
      
    ax.plot3D(xline, yline, zline, 'gray')

  
  # xline = [10, 10]
  # yline = [5, 5]
  # zline = [0, 6.67]
  temp = plot1
  for i in range(0,3):
     z =  data.loc[(data['sensor'] == 'crack') & (data['x'] == temp[0][i]) & (data['y'] == temp[1][i])]
     xline = [temp[0][i]]*2
     yline = [temp[1][i]]*2
     zline = [0, z['z']]
     ax.plot3D(xline, yline, zline, 'gray')


  ax.scatter3D(xdata, ydata, zdata, c=zdata);

def main():
  initializeGraph()
  checkNodesById()
  initializeGraphZ()
  threeDimensionGraph()

if __name__ == '__main__':
  max_velocity = 559
  min_velocity = -559
  radius_bet_two_sensor = 5
  frequency    = 100
  crack_line_node_balance = 1328  # Crack line in middle

  location = pd.read_csv('C:/Users/User/Desktop/FYP 2021/data.csv')
  data = pd.read_csv('C:/Users/User/Desktop/FYP 2021/My PZT location.csv')
  plot1 = [[], []]   # crack line
  plot2 = [[], []]
  main()
