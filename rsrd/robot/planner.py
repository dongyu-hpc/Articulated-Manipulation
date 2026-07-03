"""
Part-motion centric motion planner.
"""

from typing import Generator, Sequence
import jax
import jax.numpy as jnp
import numpy as onp
import jaxlie
from loguru import logger
import trimesh
import yourdfpy

from jaxmp import JaxKinTree
from jaxmp.coll import RobotColl, collide, Convex, link_to_capsules,  CollGeom
from jaxmp.extras.solve_ik import solve_ik
from jaxmp.robot_factors import RobotFactors


from rsrd.motion.motion_optimizer import RigidGroupOptimizer
from rsrd.robot.graspable_obj import GraspableObject
from rsrd.robot.motion_plan_yumi import YUMI_REST_POSE
from rsrd.robot.motion_plan_yumi import motion_plan_yumi


# solve_ik_vmap = jax.vmap(solve_ik, in_axes=(None, 0, None, None))
solve_ik_vmap = jax.vmap(solve_ik, in_axes=(None, 0, None, None, None, None))
mp_yumi_vmap = jax.vmap(motion_plan_yumi, in_axes=(None, None, 0))


class PartMotionPlanner:
    optimizer: RigidGroupOptimizer
    urdf: yourdfpy.URDF
    kin: JaxKinTree
    robot_coll: RobotColl
    robot_coll: RobotColl
    object: GraspableObject

    left_joint_idx: int
    right_joint_idx: int
    ee_indices: jnp.ndarray

    def __init__(
        self,
        optimizer: RigidGroupOptimizer,
        urdf: yourdfpy.URDF,
        max_motion_keyframes: int = 50,
    ):
        self.optimizer = optimizer
        self.urdf = urdf
        self.kin = JaxKinTree.from_urdf(self.urdf)

        # The convex bodies tend to be an _underestimate_ due to the decimation process.
        # To counter this, we inflate the meshes a bit.
        # def coll_handler(meshes: Sequence[trimesh.Trimesh]) -> Sequence[Convex]:
        #     _meshes = []
        #     for mesh in meshes:
        #         mesh.vertices += 0.01 * mesh.vertex_normals
        #         _meshes.append(mesh)
        #     return [Convex.from_mesh(mesh) for mesh in _meshes]
        def coll_handler(
            meshes: Sequence[trimesh.Trimesh],
            link_joint_idx: jax.Array,
            link_to_colls: dict,
        ) -> tuple:
            _meshes = []
            for mesh in meshes:
                mesh.vertices += 0.01 * mesh.vertex_normals
                _meshes.append(mesh)
            from jaxmp.coll import link_to_capsules
            return link_to_capsules(_meshes, link_joint_idx, link_to_colls)

        # Allow "collisions" within an arm. This is hardcoded for the yumi.
        # from rsrd.robot.yumi_coll_mat import self_coll_ignore_pairs
        # self.robot_coll = RobotColl.from_urdf(
        #     self.urdf, coll_handler=coll_handler, self_coll_ignore=self_coll_ignore_pairs
        # )
        # self.object = GraspableObject(self.optimizer)
        from rsrd.robot.yumi_coll_mat import self_coll_ignore_pairs
        self.robot_coll = RobotColl.from_urdf(
            self.urdf, create_coll_bodies=coll_handler, self_coll_ignore=self_coll_ignore_pairs
        )
        self.object = GraspableObject(self.optimizer)

        # the self collision matrix is incorrect??
        self.max_motion_keyframes = max_motion_keyframes

        # These TCP points are defined in `data/yumi_descriptions/yumi.urdf`.
        left_joint_name, right_joint_name = "left_dummy_joint", "right_dummy_joint"
        self.left_joint_idx = self.kin.joint_names.index(left_joint_name)
        self.right_joint_idx = self.kin.joint_names.index(right_joint_name)
        self.ee_indices = jnp.array(
            ["finger" in link_name for link_name in self.robot_coll.coll_link_names]
        )

    def plan_single(
        self, T_obj_world: jaxlie.SE3
    ) -> Generator[onp.ndarray, None, None]:
        len_traj = self.object.optimizer.part_deltas.shape[0]
        motion_subsample_rate = int(len_traj / self.max_motion_keyframes)
        timesteps = jnp.arange(len_traj, step=motion_subsample_rate)

        for part_idx, joint_idx, grasp_idx in self._get_start_cand_single(T_obj_world):
            logger.info(
                f"Trying part {part_idx}, joint {joint_idx}, with {len(grasp_idx)} trajectories."
            )
            Ts_grasp_world = self.object.get_T_grasps_world(
                part_idx, timesteps, T_obj_world
            ).wxyz_xyz[grasp_idx, None, :, :]
            logger.debug(f"Ts_grasp_world shape: {Ts_grasp_world.shape}")
            logger.debug(f"Ts_grasp_world[0, 0, 0]: {Ts_grasp_world[0, 0, 0]}")  # 第一条轨迹第一个ee第一帧
            logger.debug(f"Ts_grasp_world[0, 0, 25]: {Ts_grasp_world[0, 0, 25]}")  # 第25帧
            traj, succ = mp_yumi_vmap(
                self.kin,
                jnp.array([joint_idx]),
                Ts_grasp_world,
            )  # N_grasps, timesteps, 16

            succ_traj = traj[succ[:, 0]]

            if len(succ_traj) > 0:
                in_coll = self.get_self_coll(succ_traj).any(axis=-1)
                succ_traj = succ_traj[~in_coll]
            else:
                logger.info("No successful trajectories found.")

            if len(succ_traj) > 0:
                logger.info(f"Found {len(succ_traj)} successful trajectories.")
                succ_traj = jnp.repeat(succ_traj, motion_subsample_rate, axis=1)[
                    :, :len_traj
                ]
                yield onp.array(succ_traj)

        raise ValueError("No more valid trajectories found.")

    def plan_bimanual(
        self, T_obj_world: jaxlie.SE3
    ) -> Generator[onp.ndarray, None, None]:
        len_traj = self.object.optimizer.part_deltas.shape[0]
        motion_subsample_rate = int(len_traj / self.max_motion_keyframes)
        timesteps = jnp.arange(len_traj, step=motion_subsample_rate)

        for (
            part_idx_0,
            part_idx_1,
            joint_idx_0,
            joint_idx_1,
            grasp_idx,
        ) in self._get_start_cand_bimanual(T_obj_world):
            logger.info(
                f"Trying part {part_idx_0}, {part_idx_1}, joint {joint_idx_0}, {joint_idx_1}, with {len(grasp_idx)} trajectories."
            )
            if len(grasp_idx) == 0:
                logger.info("No valid grasps found.")

            # Batch the grasps.
            for i in range(0, grasp_idx.shape[0], 500):
                _grasp_idx = grasp_idx[i : min(i + 500, grasp_idx.shape[0])]

                logger.info(f"Attempting trajs from {len(_grasp_idx)} grasps.")
                Ts_grasp_world_0 = self.object.get_T_grasps_world(
                    part_idx_0, timesteps, T_obj_world
                ).wxyz_xyz[_grasp_idx[:, 0], None, :, :]
                Ts_grasp_world_1 = self.object.get_T_grasps_world(
                    part_idx_1, timesteps, T_obj_world
                ).wxyz_xyz[_grasp_idx[:, 1], None, :, :]
                traj, succ = mp_yumi_vmap(
                    self.kin,
                    jnp.array([joint_idx_0, joint_idx_1]),
                    jnp.concatenate([Ts_grasp_world_0, Ts_grasp_world_1], axis=1),
                )
                succ_traj = traj[succ[:, 0]]

                if len(succ_traj) > 0:
                    in_coll = self.get_self_coll(succ_traj).any(axis=-1)
                    succ_traj = succ_traj[~in_coll]
                else:
                    logger.info("No successful trajectories found.")

                if len(succ_traj) > 0:
                    logger.info(f"Found {len(succ_traj)} successful trajectories.")
                    succ_traj = jnp.repeat(succ_traj, motion_subsample_rate, axis=1)[
                        :, :len_traj
                    ]
                    yield onp.array(succ_traj)

        raise ValueError("No more valid trajectories found.")

    def _get_start_cand_single(self, T_obj_world: jaxlie.SE3):
        """
        Find a list of collision-free IK solutions to reach the object.
        """
        part_indices = self.object.rank_parts_to_move_single()

        for part_idx in part_indices:
            for joint_idx in [self.left_joint_idx, self.right_joint_idx]:
                succ_grasp_idx = self._get_coll_free_grasp_indices(
                    part_idx, joint_idx, T_obj_world
                )
                yield (part_idx, joint_idx, succ_grasp_idx)

    def _get_start_cand_bimanual(self, T_obj_world: jaxlie.SE3):
        part_indices = self.object.rank_parts_to_move_bimanual()

        grasp_idx_cache = {}  # Cache the grasp indices for each part.
        for part_idx_0, part_idx_1 in part_indices:
            for joint_idx_0, joint_idx_1 in [
                (self.left_joint_idx, self.right_joint_idx),
                (self.right_joint_idx, self.left_joint_idx)
            ]:
                if (part_idx_0, joint_idx_0) not in grasp_idx_cache:
                        grasp_idx_cache[(part_idx_0, joint_idx_0)] = self._get_coll_free_grasp_indices(
                            part_idx_0, joint_idx_0, T_obj_world
                        )
                if (part_idx_1, joint_idx_1) not in grasp_idx_cache:
                    grasp_idx_cache[(part_idx_1, joint_idx_1)] = (
                        self._get_coll_free_grasp_indices(
                            part_idx_1, joint_idx_1, T_obj_world
                        )
                    )

                # make a MxN grid of all possible combinations of grasp indices
                grasp_idx_0, grasp_idx_1 = jnp.meshgrid(
                    grasp_idx_cache[(part_idx_0, joint_idx_0)],
                    grasp_idx_cache[(part_idx_1, joint_idx_1)],
                    indexing="ij",
                )
                succ_grasp_idx = jnp.stack(
                    [grasp_idx_0.ravel(), grasp_idx_1.ravel()], axis=-1
                )

                yield (
                    part_idx_0,
                    part_idx_1,
                    joint_idx_0,
                    joint_idx_1,
                    succ_grasp_idx,
                )

    # def _get_coll_free_grasp_indices(
    #     self, part_index: int, joint_idx: int, T_obj_world: jaxlie.SE3
    # ) -> jax.Array:
    #     """
    #     Get collision-free initial (t=0) grasp indices (between the robot and the object).
    #     """
    #     T_grasp_world = self.object.get_T_grasps_world(
    #         part_index, jnp.array([0]), T_obj_world
    #     )
    #     JointVar = RobotFactors.get_var_class(self.kin, jnp.array(YUMI_REST_POSE))
    #     ik_weight = jnp.array([5.0] * 3 + [1.0] * 3)
    #     _, joints = solve_ik_vmap(
    #         self.kin,
    #         T_grasp_world,
    #         jnp.array([joint_idx]),
    #         jnp.array(YUMI_REST_POSE),
    #         JointVar,
    #         ik_weight,
    #     )

    #     # Check that the IK actually goes to the correct place.
    #     in_position = jnp.isclose(
    #         jaxlie.SE3(
    #             self.kin.forward_kinematics(joints)[..., joint_idx, :]
    #         ).translation(),
    #         T_grasp_world.translation().squeeze(),
    #         atol=0.01,
    #     ).all(axis=-1)

    #     # Collision check.
    #     robot_coll = self.robot_coll.at_joints(self.kin, joints)
    #     Ts_part_world = self.object.get_T_part_world(jnp.array(0), T_obj_world)
    #     part_coll = [
    #         Convex.from_mesh(self.object.parts[idx].mesh.convex_hull).transform(
    #             Ts_part_world[idx]
    #         )
    #         for idx in range(len(self.object.parts))
    #     ]

    #     coll_dist = jnp.zeros((*joints.shape[:-1], len(robot_coll), len(part_coll)))
    #     for i in range(len(robot_coll)):
    #         for j in range(len(part_coll)):
    #             dist = collide(robot_coll[i], part_coll[j]).dist
    #             coll_dist = coll_dist.at[..., i, j].set(dist)

    #     coll_dist = coll_dist.min(axis=-1)  # [num_joints, num_robot_links]

    #     # Make the collision check more lax for gripper ends.
    #     in_collision_non_fingers = jnp.any(
    #         coll_dist[..., ~self.ee_indices] < -0.0, axis=-1
    #     )
    #     in_collision_fingers = jnp.any(coll_dist[..., self.ee_indices] < -0.0, axis=-1)
    #     in_collision = in_collision_non_fingers | in_collision_fingers

    #     mask = ~in_collision & in_position
    #     succ_grasp_idx = jnp.where(mask)[0]

    #     return succ_grasp_idx

    def _get_coll_free_grasp_indices(
        self, part_index: int, joint_idx: int, T_obj_world: jaxlie.SE3
    ) -> jax.Array:
        T_grasp_world = self.object.get_T_grasps_world(
            part_index, jnp.array([0]), T_obj_world
        )
        JointVar = RobotFactors.get_var_class(self.kin, jnp.array(YUMI_REST_POSE))
        ik_weight = jnp.array([5.0] * 3 + [1.0] * 3)

        _, joints = solve_ik_vmap(
            self.kin,
            T_grasp_world,
            jnp.array([joint_idx]),
            jnp.array(YUMI_REST_POSE),
            JointVar,
            ik_weight,
        )

        # Check that the IK actually goes to the correct place.
        in_position = jnp.isclose(
            jaxlie.SE3(
                self.kin.forward_kinematics(joints)[..., joint_idx, :]
            ).translation(),
            T_grasp_world.translation().squeeze(),
            atol=0.01,
        ).all(axis=-1)

        # ↓ 加这几行
        logger.debug(f"  Part {part_index}, joint {joint_idx}: {T_grasp_world.translation().shape[0]} grasps total")
        logger.debug(f"  IK in_position: {in_position.sum()} / {len(in_position)}")


        # Collision check: robot vs object parts.
        Ts_part_world = self.object.get_T_part_world(jnp.array(0), T_obj_world)
        part_coll = [
            Convex.from_mesh(self.object.parts[idx].mesh.convex_hull).transform(
                Ts_part_world[idx]
            )
            for idx in range(len(self.object.parts))
        ]

        # Use batched CollGeom: robot_coll shape is (*batch, num_links)
        robot_coll = self.robot_coll.at_joints(self.kin, joints)
        assert isinstance(robot_coll, CollGeom)
        num_links = self.robot_coll.num_colls

        coll_dist = jnp.full((*joints.shape[:-1], num_links, len(part_coll)), jnp.inf)
        for j, pc in enumerate(part_coll):
            dist = collide(robot_coll, pc).dist  # (*batch, num_links) (480, 21)
            logger.debug(f"  collide dist shape: {dist.shape}, min: {dist.min():.4f}, max: {dist.max():.4f}")
            # break 
            coll_dist = coll_dist.at[..., j].set(dist) # 写入 (480, 21, j)

        coll_dist = coll_dist.min(axis=-1)  # (*batch, num_links) 在 part_coll 维度上取 min → (480, 21)

        in_collision_non_fingers = jnp.any(
            coll_dist[..., ~self.ee_indices] < -0.05, axis=-1
        )
        in_collision_fingers = jnp.any(coll_dist[..., self.ee_indices] < -0.05, axis=-1)
        in_collision = in_collision_non_fingers | in_collision_fingers

        logger.debug(f"  in_collision: {in_collision.sum()} / {len(in_collision)}")
        logger.debug(f"  final valid: {(~in_collision & in_position).sum()}")

        mask = ~in_collision & in_position
        return jnp.where(mask)[0]


    def get_self_coll(self, joints: jnp.ndarray) -> jnp.ndarray:
        """
        Get self-collision distances for the robot.
        """
        # robot_coll = self.robot_coll.at_joints(self.kin, joints)
        # assert isinstance(robot_coll, list)

        # coll_dist = jnp.zeros((*joints.shape[:-1], len(robot_coll), len(robot_coll)))
        # for i in range(len(robot_coll)):
        #     for j in range(len(robot_coll)):
        #         dist = collide(robot_coll[i], robot_coll[j]).dist
        #         coll_dist = coll_dist.at[..., i, j].set(dist)

        # return jnp.any(
        #     coll_dist * self.robot_coll.self_coll_mat < 0.0, axis=(-1, -2)
        # )
        dist = self.robot_coll.self_coll_dist(self.kin, joints)  # (*batch,)
        return dist < 0.0

    def get_approach_motion(
        self, start_joints: jnp.ndarray, approach_dist: float
    ) -> jnp.ndarray:
        """
        Plan an approach motion to the object, with an approach direction.
        """
        assert start_joints.shape == (self.kin.num_actuated_joints,)
        Ts_joint_world = self.kin.forward_kinematics(start_joints)
        z_axis = jaxlie.SE3(Ts_joint_world).rotation().as_matrix()[:, :, 2]
        Ts_approach = Ts_joint_world.at[..., 4:].set(
            Ts_joint_world[..., 4:] - approach_dist * z_axis
        )

        joint_idx = jnp.array([self.left_joint_idx, self.right_joint_idx])
        Ts_approach = Ts_approach[joint_idx]
        JointVar = RobotFactors.get_var_class(self.kin, start_joints)
        ik_weight = jnp.array([5.0] * 3 + [1.0] * 3)

        # _, joints = solve_ik(
        #     self.kin,
        #     jaxlie.SE3(Ts_approach),
        #     joint_idx,
        #     start_joints,  # Keep it close to the start-of-traj joints.
        #     pos_weight=5.0,
        #     rot_weight=1.0,
        #     limit_weight=100.0,
        #     joint_vel_weight=100.0,  # No joint twists.
        # )
        _, joints = solve_ik(
            self.kin,
            jaxlie.SE3(Ts_approach),
            joint_idx,
            start_joints,
            JointVar,
            ik_weight,
            limit_weight=100.0,
            joint_vel_weight=100.0,
        )

        traj_rest_approach = jnp.array(YUMI_REST_POSE)[None, :] + jnp.linspace(
            0, 1, 20
        )[:, None] * (joints - jnp.array(YUMI_REST_POSE)[None, :])
        traj_approach_start = jnp.array(joints)[None, :] + jnp.linspace(0, 1, 10)[
            :, None
        ] * (start_joints[None, :] - jnp.array(joints)[None, :])

        return jnp.concatenate([traj_rest_approach, traj_approach_start], axis=0)