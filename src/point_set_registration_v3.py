import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        # zoom=0.4459,
        # front=[0.9288, -0.2951, -0.2242],
        # lookat=[1.6784, 2.0612, 1.4451],
        # up=[-0.3402, -0.9189, -0.1996],
    )


demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud("./cube.pcd")
target = o3d.io.read_point_cloud("./cube2.pcd")
threshold = 0.02
trans_init = np.asarray(
    [
        [1.5, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
draw_registration_result(source, target, trans_init)
target_transform = target.transform(trans_init)


print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source,
    target,
    threshold,
    trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
)
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
transformation = reg_p2p.transformation
transformation_inverse = np.linalg.inv(transformation)
target_transform = target_transform.transform(transformation_inverse)

draw_registration_result(
    source,
    target_transform,
    [
        [1.0001, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)
