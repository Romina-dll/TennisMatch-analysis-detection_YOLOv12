"""
Calculate the center coordinates of a bounding box.    
Args:
    bbox (list or tuple): Bounding box coordinates [x1, y1, x2, y2]        
Returns:
    tuple: (center_x, center_y) coordinates of the bounding box center
"""
def get_center_bbox(bbox):
    x1,y1,x2,y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x,center_y)

"""
Compute the Euclidean distance between two points in 2D space.
Args:
    p1 (tuple): First point coordinates (x1, y1)
    p2 (tuple): Second point coordinates (x2, y2)        
Returns:
    float: Euclidean distance between p1 and p2
"""
def measure_distance(p1,p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1])**2)**0.5