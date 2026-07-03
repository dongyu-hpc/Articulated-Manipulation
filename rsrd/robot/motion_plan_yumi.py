"""Motion planning implementation for Yumi, in JAX.

Requires jaxfg:

    conda install -c conda-forge suitesparse

    cd ~
    git clone https://github.com/brentyi/jaxfg.git
    cd jaxfg
    pip install -e .
"""

from __future__ import annotations

import jax
import jaxlie
import jax_dataclasses as jdc
from jax import Array, numpy as jnp
from jaxmp import RobotFactors, JaxKinTree
import jaxls
from jaxtyping import Float

YUMI_REST_POSE = [
    1.21442839,
    -1.03205606,
    -1.10072738,
    0.2987352,
    -1.85257716,
    1.25363652,
    -2.42181893,
    -1.24839656,
    -1.09802876,
    1.06634394,
    0.31386161,
    1.90125141,
    1.3205139,
    2.43563939,
    0.025,
    0.025,
]

@jdc.jit
def motion_plan_yumi(
    kin: JaxKinTree,
    target_joint_inds: Float[Array, "inds"],
    target_ee_per_ind: Float[Array, "inds timestep 7"],
    rest_pose: Array = jnp.array(YUMI_REST_POSE),
    pos_weight: jdc.Static[float] = 5.0,
    rot_weight: jdc.Static[float] = 1.0,
    limit_weight: jdc.Static[float] = 100.0,
    rest_weight: jdc.Static[float] = 0.01,
    smoothness_weight: jdc.Static[float] = 1.0,
) -> tuple[Array, Array]:
    num_inds, timesteps, pose_dim = target_ee_per_ind.shape
    assert pose_dim == 7

    JointVar = RobotFactors.get_var_class(kin, default_val=rest_pose)
    ik_weight = jnp.array([pos_weight] * 3 + [rot_weight] * 3)

    # target_ee_per_ind: (inds, timestep, 7) → per-timestep SE3 for each ee
    # 转成 (timestep, inds, 7) 方便按 timestep 索引
    target_poses = jaxlie.SE3(
        jnp.swapaxes(target_ee_per_ind, 0, 1)
    )  # (timestep, inds, 7)

    def ik_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jax.Array],
        target: jaxlie.SE3,
    ) -> jax.Array:
        joint_cfg = vals[joint_var]
        Ts_joint_world = kin.forward_kinematics(joint_cfg)
        residual = (
            jaxlie.SE3(Ts_joint_world[target_joint_inds]).inverse()
            @ target
        ).log()
        return (residual * ik_weight).flatten()

    factors = [
        jaxls.Factor(
            ik_cost,
            (
                JointVar(jnp.arange(timesteps)),
                target_poses,  # (timestep, inds, 7) 批量传入
            ),
        ),
        RobotFactors.limit_cost_factor(
            JointVar,
            jnp.arange(timesteps),
            kin,
            jnp.array([limit_weight] * kin.num_actuated_joints),
        ),
        RobotFactors.rest_cost_factor(
            JointVar,
            jnp.arange(timesteps),
            jnp.array([rest_weight] * kin.num_actuated_joints),
        ),
        RobotFactors.smoothness_cost_factor(
            JointVar,
            jnp.arange(1, timesteps),
            jnp.arange(0, timesteps - 1),
            jnp.array([smoothness_weight] * kin.num_actuated_joints),
        ),
    ]

    traj_vars = [JointVar(jnp.arange(timesteps))]
    graph = jaxls.FactorGraph.make(factors, traj_vars, use_onp=False)
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(traj_vars),
        linear_solver="conjugate_gradient",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5, parameter_tolerance=1e-5, max_iterations=50,
        ),
        verbose=False,
    )
    traj = jnp.stack([solution[JointVar(tstep)] for tstep in range(timesteps)])

    # Check success
    def check_success(traj, target_joint_inds, target_ee_per_ind, kin):
        def body_fun(i, success):
            idx = target_joint_inds[i]
            pose_from_joints = kin.forward_kinematics(traj)[:, idx]
            pose_target = target_ee_per_ind[i]
            succ = jnp.all(
                jnp.isclose(
                    jaxlie.SE3(pose_target).translation(),
                    jaxlie.SE3(pose_from_joints).translation(),
                    atol=1e-2,
                )
            )
            return jnp.logical_and(success, succ)

        success = jnp.ones((), dtype=bool)
        success = jax.lax.fori_loop(0, len(target_joint_inds), body_fun, success)
        return success

    success = check_success(traj, target_joint_inds, target_ee_per_ind, kin)
    return traj, jnp.array([success])


# @jdc.jit
# def motion_plan_yumi(
#     kin: JaxKinTree,
#     target_joint_inds: Float[Array, "inds"],
#     target_ee_per_ind: Float[Array, "inds timestep 7"],
#     rest_pose: Array = jnp.array(YUMI_REST_POSE),
#     pos_weight: jdc.Static[float] = 5.0,
#     rot_weight: jdc.Static[float] = 1.0,
#     limit_weight: jdc.Static[float] = 100.0,
#     rest_weight: jdc.Static[float] = 0.01,
#     smoothness_weight: jdc.Static[float] = 1.0,
# ) -> tuple[Array, Array]:
#     _, timesteps, pose_dim = target_ee_per_ind.shape
#     assert pose_dim == 7

#     # Create factor graph.
#     JointVar = RobotFactors.get_var_class(kin, default_val=rest_pose)
#     traj_vars = [JointVar(id=i) for i in range(timesteps)]

#     factors = []
#     ik_weight = jnp.array([pos_weight] * 3 + [rot_weight] * 3)
    
#     for tstep in range(timesteps):
#         for i, idx in enumerate(target_joint_inds):
#             factors.append(
#                 RobotFactors.ik_cost_factor(
#                     JointVar,
#                     tstep,
#                     kin,
#                     jaxlie.SE3(target_ee_per_ind[i][tstep]),
#                     jnp.array([idx]),
#                     ik_weight,
#                 )
#             )
        
#         factors.extend([
#             RobotFactors.limit_cost_factor(
#                 JointVar,
#                 tstep,
#                 kin,
#                 jnp.array([limit_weight] * kin.num_actuated_joints),
#             ),
#             RobotFactors.rest_cost_factor(
#                 JointVar,
#                 tstep,
#                 jnp.array([rest_weight] * kin.num_actuated_joints),
#             ),
#         ])
        
#         if tstep > 0:
#             factors.append(
#                 RobotFactors.smoothness_cost_factor(
#                     JointVar,
#                     tstep,
#                     tstep - 1,
#                     jnp.array([smoothness_weight] * kin.num_actuated_joints),
#                 )
#             )

#     graph = jaxls.FactorGraph.make(
#         factors,
#         traj_vars,
#         use_onp=False,
#     )
#     solution = graph.solve(
#         initial_vals=jaxls.VarValues.make(traj_vars),
#         linear_solver="conjugate_gradient",
#         trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
#         termination=jaxls.TerminationConfig(
#             gradient_tolerance=1e-5, parameter_tolerance=1e-5, max_iterations=50,
#         ),
#         verbose=False,
#     )
#     traj = jnp.array([solution[var] for var in traj_vars])

#     # Check success for all joint indices.
#     def check_success(traj, target_joint_inds, target_ee_per_ind, kin):
#         def body_fun(i, success):
#             idx = target_joint_inds[i]
#             pose_from_joints = kin.forward_kinematics(traj)[:, idx]
#             pose_target = target_ee_per_ind[i]
#             succ = jnp.all(
#                 jnp.isclose(
#                     jaxlie.SE3(pose_target).translation(),
#                     jaxlie.SE3(pose_from_joints).translation(),
#                     atol=1e-2
#                 )
#             )
#             return jnp.logical_and(success, succ)

#         success = jnp.ones((), dtype=bool)
#         success = jax.lax.fori_loop(0, len(target_joint_inds), body_fun, success)
#         return success

#     success = check_success(traj, target_joint_inds, target_ee_per_ind, kin)

#     return traj, jnp.array([success])

# def motion_plan_yumi(
#     kin: JaxKinTree,
#     target_joint_inds: Float[Array, "inds"],
#     target_ee_per_ind: Float[Array, "inds timestep 7"],
#     rest_pose: Array = jnp.array(YUMI_REST_POSE),
#     pos_weight: jdc.Static[float] = 5.0,
#     rot_weight: jdc.Static[float] = 1.0,
#     limit_weight: jdc.Static[float] = 100.0,
#     rest_weight: jdc.Static[float] = 0.01,
#     smoothness_weight: jdc.Static[float] = 1.0,
# ) -> tuple[Array, Array]:
#     _, timesteps, pose_dim = target_ee_per_ind.shape
#     assert pose_dim == 7

#     # Create factor graph.
#     JointVar = RobotFactors.get_var_class(kin, default_val=rest_pose)
#     traj_vars = [JointVar(id=i) for i in range(timesteps)]

#     factors = []
#     for tstep in range(timesteps):
#         for i, idx in enumerate(target_joint_inds):
#             factors.extend(
#                 [
#                     jaxls.Factor.make(
#                         RobotFactors.ik_cost,
#                         (
#                             kin,
#                             traj_vars[tstep],
#                             jaxlie.SE3(target_ee_per_ind[i][tstep]),
#                             idx,
#                             jnp.array([pos_weight] * 3 + [rot_weight] * 3),
#                         ),
#                     ),
#                 ]
#             )
#         factors.extend(
#             [
#                 jaxls.Factor.make(
#                     RobotFactors.limit_cost,
#                     (
#                         kin,
#                         traj_vars[tstep],
#                         jnp.array([limit_weight] * kin.num_actuated_joints),
#                     ),
#                 ),
#                 jaxls.Factor.make(
#                     RobotFactors.rest_cost,
#                     (
#                         traj_vars[tstep],
#                         jnp.array([rest_weight] * kin.num_actuated_joints),
#                     ),
#                 ),
#             ]
#         )
#         if tstep > 0:
#             factors.append(
#                 jaxls.Factor.make(
#                     RobotFactors.smoothness_cost,
#                     (
#                         traj_vars[tstep],
#                         traj_vars[tstep - 1],
#                         jnp.array([smoothness_weight] * kin.num_actuated_joints),
#                     ),
#                 )
#             )

#     graph = jaxls.FactorGraph.make(
#         factors,
#         traj_vars,
#         use_onp=False,
#     )
#     solution = graph.solve(
#         initial_vals=jaxls.VarValues.make(traj_vars),
#         linear_solver="conjugate_gradient",
#         trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
#         termination=jaxls.TerminationConfig(
#             gradient_tolerance=1e-5, parameter_tolerance=1e-5, max_iterations=50,
#         ),
#         verbose=False,
#     )
#     traj = jnp.array([solution[var] for var in traj_vars])

#     # Check success for all joint indices.
#     def check_success(traj, target_joint_inds, target_ee_per_ind, kin):
#         def body_fun(i, success):
#             idx = target_joint_inds[i]
#             pose_from_joints = kin.forward_kinematics(traj)[:, idx]
#             pose_target = target_ee_per_ind[i]
#             succ = jnp.all(
#                 jnp.isclose(
#                     jaxlie.SE3(pose_target).translation(),
#                     jaxlie.SE3(pose_from_joints).translation(),
#                     atol=1e-2
#                 )
#             )
#             return jnp.logical_and(success, succ)

#         success = jnp.ones((), dtype=bool)
#         success = jax.lax.fori_loop(0, len(target_joint_inds), body_fun, success)
#         return success

#     success = check_success(traj, target_joint_inds, target_ee_per_ind, kin)

#     return traj, jnp.array([success])