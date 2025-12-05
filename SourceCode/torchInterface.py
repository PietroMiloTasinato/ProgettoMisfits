import open3d as o3d
import numpy as np
import time


def load_pcd_file(file_path): #Path to the .pcd file

    #Load a PCD file and display basic information about the pointcloud.
    try:
        # Load the pointcloud
        pcd = o3d.io.read_point_cloud(file_path)
        
        # Display basic information
        print(f"pointcloud loaded successfully from: {file_path}")
        print(f"Number of points: {len(pcd.points)}")
        print(f"Has normals: {pcd.has_normals()}")
        print(f"Has colors: {pcd.has_colors()}")
        
        # Get the bounding box
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            print(f"\nBounding box:")
            print(f"  Min: {points.min(axis=0)}")
            print(f"  Max: {points.max(axis=0)}")
        
        return pcd
        
    except Exception as e:
        print(f"Error loading PCD file: {e}")
        return None

def visualize_pcd(pcd): #The pointcloud to visualize
    
    
    #Visualize the pointcloud in a 3D window.
    if pcd is not None and len(pcd.points) > 0:
        print("\nOpening window")
        o3d.visualization.draw_geometries([pcd], window_name="pointcloud Viewer", width=800, height=600)
        time.sleep(1)
    else:
        print("Invalid pointcloud")

def main():

    file_path = "C:/Users/pietr/Desktop/ProgettoMisfits-main/Materiale_challange_2/Vineyard Pointcloud/dataset/pointcloud/pc_color_filtered.pcd"
    
    # Load the pointcloud
    point_cloud = load_pcd_file(file_path)
    
    # Visualize the pointcloud
    if point_cloud is not None:
        visualize_pcd(point_cloud)

    main()