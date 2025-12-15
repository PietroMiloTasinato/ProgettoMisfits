import open3d as o3d
import numpy as np
from scipy.interpolate import splprep, splev
import networkx as ntx

### this prohgram takes in a pointcloud of a plant or other tree-like structures and returns a networkx graph 
#   rappresentation of the pointcloud, with a node on the trunk and each termination of the branches, 
#   if a node is connected to more than 1 other node, then it's a bifurcation
###

# function to extract a good starting point for costructing the graph, while evaiding noise and the ground, 
# it takes in the pointcloud as an open3d pointcloud, the radius for the search and the number of nearest neighbor (hyperparameter) as input. 
# It then returns the index of that starting point, if no points are eligible, returns None

class FloatList:
    """A class for managing a list of floating-point numbers."""
    
    def __init__(self, values=None):
        """Initialize with an optional list of floats."""
        self.__data = list(values) if values else []
    
    def add(self, value):
        """Add a single float to the list."""
        self.__data.append(float(value))
    
    def extend(self, values):
        """Add multiple floats to the list."""
        self.__data.extend([float(v) for v in values])
    
    def get(self, index):
        """Get the float at the specified index."""
        return self.__data[index]
    
    def set(self, index, value):
        """Set the float at the specified index."""
        self.__data[index] = float(value)
    
    def remove(self, index):
        """Remove and return the float at the specified index."""
        return self.__data.pop(index)
    
    def clear(self):
        """Remove all elements from the list."""
        self.__data.clear()
    
    def size(self):
        """Return the number of elements in the list."""
        return len(self.__data)
    
    def is_empty(self):
        """Check if the list is empty."""
        return len(self.__data) == 0
    
    def sum(self):
        """Return the sum of all floats."""
        return sum(self.__data)
    
    def mean(self):
        """Return the average of all floats."""
        if not self.__data:
            return 0.0
        return sum(self.__data) / len(self.__data)
    
    def min(self):
        """Return the minimum value."""
        return min(self.__data) if self.__data else None
    
    def max(self):
        """Return the maximum value."""
        return max(self.__data) if self.__data else None
    
    def sort(self, reverse=False):
        """Sort the list in place."""
        self.__data.sort(reverse=reverse)
    
    def filter(self, predicate):
        """Return a new FloatList with elements that satisfy the predicate."""
        return FloatList([x for x in self.__data if predicate(x)])
    
    def map(self, func):
        """Return a new FloatList with the function applied to each element."""
        return FloatList([func(x) for x in self.__data])
    
    def to_list(self):
        """Return a copy of the internal list."""
        return self.__data.copy()
    
    def __str__(self):
        """String representation of the FloatList."""
        return f"FloatList({self.__data})"
    
    def __repr__(self):
        """Official string representation."""
        return f"FloatList({self.__data})"
    
    def __len__(self):
        """Support for len() function."""
        return len(self.__data)
    
    def __getitem__(self, index):
        """Support for indexing with []."""
        return self.__data[index]
    
    def __setitem__(self, index, value):
        """Support for assignment with []."""
        self.__data[index] = float(value)
    
    def __iter__(self):
        """Support for iteration."""
        return iter(self.__data)



def advance_along_curve(tck, u0, ds, du=1e-6):
    """
    Advance along a parametric curve by arc-length ds.
    
    tck : spline representation
    u0  : starting parameter
    ds  : desired distance
    du  : small parameter step
    """
    p0 = np.array(splev(u0, tck))
    dist = 0.0
    u = u0

    while dist < ds and u < 1.0:
        u_next = min(u + du, 1.0)
        p1 = np.array(splev(u_next, tck))
        dist += np.linalg.norm(p1 - p0)
        p0 = p1
        u = u_next

    return p0


def union(pcd_tree, nextPoint, knownPoints, unionDistance):
    flag = False
    
    points = pcd_tree.search_radius_vector_3d(nextPoint, unionDistance)
    
    for i in len(points):
        for j in len(knownPoints):
            if(points[i][0] == knownPoints[j][0] and points[i][1] == knownPoints[j][1] and points[i][2] == knownPoints[j][2]): flag = True
        
    
    return flag


def startPoint(pcd: o3d.geometry.PointCloud, radius, nnNum):
    
    pcd = pcd.remove_duplicated_points()
    
    startPoints = []
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    flag, tot = (0,)*2
    
    for p in pcd.points:
        pointArray = pcd_tree.search_radius_vector_3d(p, radius) # pointArray = [number of points founsd, [vector of indexes], [distance from p for each point]]
        #print(pointArray)
        j = 1
        
        if(pointArray[0] > 9):
            for i in pointArray[1]:
                for k in range(j, len(pointArray[1])):
                    tot += 1
                    if(np.sqrt(np.square(pcd.points[i][0] - pcd.points[k][0]) + np.square(pcd.points[i][1] - pcd.points[k][1]) + np.square(pcd.points[i][2] - pcd.points[k][2])) <= radius): 
                        flag += 1
                        print("one") 
                    
                j += 1
            
            if((flag/tot) >= nnNum): 
                startPoints.append(p)
                
            tot = 0
            flag = 0
    
    return startPoints


def buildGraph(pcd: o3d.geometry.PointCloud, startPoints, explLength = 0.03, unionDistance = 0.01): #provare con esploration distance 0.05 e union dist 0.03
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    finalGraph = ntx.Graph()
    maxGraph = []
    initRot = np.identity(3)
    centers = []
    knownPoints = []
    
    print("begin: " + str(len(startPoints)))
    
    for p in startPoints:
        
        print("kick")
        
        graph = []
        inliers_points = []
        
        for i in range(11):
            
            extent = np.array([(i)*explLength/10, (i)*explLength/10, (i)*explLength/10])
            obb = o3d.geometry.OrientedBoundingBox(p,initRot,extent)
            inliers_indices = obb.get_point_indices_within_bounding_box(pcd.points)
            
            for j in inliers_indices:
                inliers_points.append(pcd.points[j])
            
            centers.append(np.mean(inliers_points))
            
        centers = np.transpose(np.asanyarray(centers))
        x = centers[0]
        y=centers[1]
        z=centers[2]

        # Fit exact spline through all points
        tck, u = splprep([x, y, z], s=0)
        # Sample the curve
        u_fine = np.linspace(0, 1, 200)
        x_new, y_new, z_new = splev(u_fine, tck)
        
        u0 = 0.5
        P = np.array(splev(u0, tck))
        T = np.array(splev(u0, tck, der=1))
        T = T / np.linalg.norm(T)
        a, b, c = T
        d = -np.dot(T, P)
        in_plane = []
        
        for i in inliers_points:
            if(a*i[0]+b*i[1]+c*i[2]+d == 0): in_plane.append(i)
        if(in_plane == []): graph.append(0.000001)
        
        thikcness = 0
        j = 0
        
        for i in in_plane[1]:
                for k in range(j, len(in_plane[1])):
                    if(np.sqrt(np.square(in_plane[i][0] - in_plane[k][0]) + np.square(in_plane[i][1] - in_plane[k][1]) + np.square(in_plane[i][2] - in_plane[k][2])) > thikcness): thikcness = np.sqrt(np.square(in_plane[i][0] - in_plane[k][0]) + np.square(in_plane[i][1] - in_plane[k][1]) + np.square(in_plane[i][2] - in_plane[k][2]))
                j += 1
        
        graph.append(thikcness)
        
        nextPoint = advance_along_curve(tck, u[-1], ds=0.05)
        knownPoints.append(nextPoint)
        
        nextPoints = [centers[-1], nextPoint, np.asarray([nextPoint[0], nextPoint[1], nextPoint[2] + 0.025]), np.asarray([nextPoint[0], nextPoint[1]+ 0.025, nextPoint[2]])]
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(nextPoints)
        centers = []
        
        while((not obb.is_empty()) and (not union(pcd_tree, nextPoint, unionDistance, knownPoints))):
            
            print("second loop")
            
            inliers_indices = obb.get_point_indices_within_bounding_box(pcd.points)
            
            for j in inliers_indices:
                inliers_points.append(pcd.points[j])
    
            centers.append(np.mean(inliers_points))
            
            x, y, z = centers.T

            # Fit exact spline through all points
            tck, u = splprep([x, y, z], s=0)
            # Sample the curve
            u_fine = np.linspace(0, 1, 200)
            x_new, y_new, z_new = splev(u_fine, tck)
        
            u0 = 0.5
            P = np.array(splev(u0, tck))
            T = np.array(splev(u0, tck, der=1))
            T = T / np.linalg.norm(T)
            a, b, c = T
            d = -np.dot(T, P)
            in_plane = []
        
            for i in inliers_points:
                if(a*i[0]+b*i[1]+c*i[2]+d == 0): in_plane.append(i)
            if(in_plane == []): graph.append(0.000001)
        
            thikcness = 0
            j = 0
        
            for i in in_plane[1]:
                    for k in range(j, len(in_plane[1])):
                        if(np.sqrt(np.square(in_plane[i][0] - in_plane[k][0]) + np.square(in_plane[i][1] - in_plane[k][1]) + np.square(in_plane[i][2] - in_plane[k][2])) > thikcness): thikcness = np.sqrt(np.square(in_plane[i][0] - in_plane[k][0]) + np.square(in_plane[i][1] - in_plane[k][1]) + np.square(in_plane[i][2] - in_plane[k][2]))
                    j += 1
            
            graph.append(thikcness)
            
            nextPoint = advance_along_curve(tck, u[-1], ds=0.05)
            knownPoints.append(nextPoint)
            
            nextPoints = [centers[-1], nextPoint, np.asarray([nextPoint[0], nextPoint[1], nextPoint[2] + 0.025]), np.asarray([nextPoint[0], nextPoint[1]+ 0.025, nextPoint[2]])]
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(nextPoints)
            centers = []
        
        if(len(graph)> len(maxGraph)): maxGraph = graph.copy()
        
    
    finalGraph.add_nodes_from(maxGraph)
    finalGraph.add_edges_from(zip(maxGraph[:-1], maxGraph[1:]))
        
    return finalGraph



def main():
    
    file_path = "C:/Users/pietr/Desktop/ProgettoMisfits-main/Materiale_challange_2/Vineyard Pointcloud/dataset/pointcloud/pc_color_filtered.pcd"
    pcd = o3d.io.read_point_cloud(file_path)
    
    startPoints = startPoint(pcd, 0.005, 0.1)
    
    print(len(startPoints))
    
    print(buildGraph(pcd, np.asanyarray(startPoints)))
    
main()
    