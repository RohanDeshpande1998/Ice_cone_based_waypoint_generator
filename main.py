
import math
import numpy as np

def generate_waypoint(robot_motion_vector, goal, sensor_data_points):
    """_summary_

    Args:
        robot_motion_vector (vector3d): vector defining where the robot is going to move
        goal (vector3d): goal vector is defined w.r.t robot
        sensor_data_points (array of vector3d): PointCloud coordinate data w.r.t robot

    Returns:
        waypoint (vector3d): vector from robot frame of reference
        cone_axis_angle (float): cone angle in radians
        sphere_radius (float): radius of sphere that touches the cone
    """
    robot_motion_vector_hat = robot_motion_vector/np.linalg.norm(robot_motion_vector)
    robot_to_sphere_center_distance_candidates = []
    cone_axis_angle_candidates = []
    cone_axis_candidates = []

    sensor_data_point_directions = sensor_data_points/np.linalg.norm(sensor_data_points, axis=1, keepdims=True)
    sensor_data_point_distances = np.linalg.norm(sensor_data_points, axis=1)
    for direction in sensor_data_point_directions:
        dot_products = np.clip(np.dot(sensor_data_point_directions, direction), -1.0, 1.0)
        valid_mask_dot_products_non_negative = dot_products >= 0
        valid_mask_dot_product_less_than_axis_angle = dot_products >= np.dot(direction,robot_motion_vector_hat)
        valid_points_mask = valid_mask_dot_products_non_negative & valid_mask_dot_product_less_than_axis_angle
        if np.any(valid_points_mask):
            valid_sensor_data_point_directions = sensor_data_point_directions[valid_points_mask]
            valid_sensor_data_point_distances = sensor_data_point_distances[valid_points_mask]
            valid_dot_products = dot_products[valid_points_mask]
            A = np.linalg.norm(np.cross(direction, robot_motion_vector_hat))**2
            B = np.linalg.norm(np.cross(direction, valid_sensor_data_point_directions), axis = 1)**2
            C = valid_dot_products + np.sqrt(abs(A - B))
            D = valid_sensor_data_point_distances/C
            robot_to_sphere_center_distance_candidates.append(np.min(D))
            cone_axis_angle_candidates.append(np.arccos(np.dot(direction,robot_motion_vector_hat)))
            cone_axis_candidates.append(direction)

    cone_axis_candidates = np.array(cone_axis_candidates)
    cone_axis_angle_candidates = np.array(cone_axis_angle_candidates)
    robot_to_sphere_center_distance_candidates = np.array(robot_to_sphere_center_distance_candidates)
    d = 0

    waypoint_candidates = cone_axis_candidates*robot_to_sphere_center_distance_candidates[:, np.newaxis]
    min_dist = math.inf
    for index, possible_waypoint in enumerate(waypoint_candidates):
        curr_dist = np.linalg.norm(possible_waypoint - goal)
        if curr_dist < min_dist:
            min_dist = curr_dist
            waypoint = list(possible_waypoint)
            cone_axis_angle = cone_axis_angle_candidates[index]
            d = robot_to_sphere_center_distance_candidates[index]
            cone_axis = cone_axis_candidates[index]
    sphere_radius = d * math.sin(cone_axis_angle)

    return waypoint, cone_axis_angle, sphere_radius 
    




def main():
    points = np.array([
        [1, 2, 3],
        [-2, -1, 4],
        [3, -3, 2],
        [-1, -2, -3],
        [3,3,0],
        [3,0,0]
    ])
    goal = np.array([1, 1, 1])
    robot_motion_vector = np.array([-1,3,0])

    generate_waypoint(robot_motion_vector,goal,points)

if __name__ == "__main__":
    main()
