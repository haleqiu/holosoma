"""State recorder for logging proprioceptive and IMU data during MuJoCo simulation.

This module provides a simple interface to record robot state data during simulation,
including joint positions/velocities and IMU measurements (orientation, angular velocity,
linear acceleration).

Output format is pickle (.pkl) compatible with locomotion_dataset.pkl structure.

Example usage:
    >>> from holosoma.utils.state_recorder import StateRecorder
    >>> recorder = StateRecorder(simulator)
    >>> # In simulation loop:
    >>> recorder.record()
    >>> # After simulation:
    >>> recorder.save("robot_data.pkl")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


@dataclass
class StateRecorderConfig:
    """Configuration for state recorder.

    Attributes
    ----------
    env_id : int
        Environment ID to record (for multi-env simulations).
    record_torques : bool
        Whether to record applied torques.
    sequence_name : str
        Name for the recorded sequence.
    """

    env_id: int = 0
    record_torques: bool = True
    sequence_name: str = "mujoco_simulation"


@dataclass
class RecordedData:
    """Container for recorded simulation data.

    All arrays have shape [num_steps, ...] where num_steps is the actual
    number of recorded steps.

    Data Shapes:
    ------------
    - timestamps: [num_steps]
    - dof_pos: [num_steps, num_dof] - joint positions (rad)
    - dof_vel: [num_steps, num_dof] - joint velocities (rad/s)
    - torques: [num_steps, num_dof] or None - torques (Nm)
    - base_quat: [num_steps, 4] - base orientation (xyzw format internally)
    - base_angular_vel: [num_steps, 3] - base angular velocity (rad/s)
    - base_linear_acc: [num_steps, 3] - base linear acceleration (m/s²)
    - root_pos: [num_steps, 3] - root position (m)
    - root_lin_vel: [num_steps, 3] - root linear velocity (m/s)
    - torso_quat: [num_steps, 4] or None - torso orientation (xyzw format internally, MuJoCo only)
    - torso_angular_vel: [num_steps, 3] or None - torso angular velocity (rad/s, MuJoCo only)
    - torso_linear_acc: [num_steps, 3] or None - torso linear acceleration (m/s², MuJoCo only)
    - commands: [num_steps, command_dim] or None - locomotion commands (vx, vy, yaw_rate, etc.)
    - motor_cmd: [num_steps, num_motors, 5] or None - motor commands [q, dq, tau, kp, kd] (bridge system)
    """

    # Timestamps
    timestamps: np.ndarray  # [num_steps]

    # Proprioceptive data
    dof_pos: np.ndarray  # [num_steps, num_dof]
    dof_vel: np.ndarray  # [num_steps, num_dof]
    torques: np.ndarray | None  # [num_steps, num_dof] or None

    # IMU data (internal format: xyzw, matching holosoma convention)
    base_quat: np.ndarray  # [num_steps, 4] (xyzw format, converted to wxyz when saved)
    base_angular_vel: np.ndarray  # [num_steps, 3]
    base_linear_acc: np.ndarray  # [num_steps, 3]

    # Root state (position + orientation + velocities)
    root_pos: np.ndarray  # [num_steps, 3]
    root_lin_vel: np.ndarray  # [num_steps, 3]

    # Torso IMU data (MuJoCo only, optional, internal format: xyzw)
    torso_quat: np.ndarray | None = None  # [num_steps, 4] xyzw format (converted to wxyz when saved)
    torso_angular_vel: np.ndarray | None = None  # [num_steps, 3]
    torso_linear_acc: np.ndarray | None = None  # [num_steps, 3]

    # Commands (policy/locomotion commands)
    commands: np.ndarray | None = None  # [num_steps, command_dim] locomotion commands (vx, vy, yaw_rate, walk_stand, etc.)

    # Motor commands (low-level motor commands from bridge system)
    # Shape: [num_steps, num_motors, 5] where 5 = [q, dq, tau, kp, kd]
    #   - q: desired joint position (rad)
    #   - dq: desired joint velocity (rad/s)
    #   - tau: feedforward torque (Nm)
    #   - kp: position gain
    #   - kd: velocity gain
    motor_cmd: np.ndarray | None = None

    # Metadata
    dof_names: list[str] = field(default_factory=list)
    sim_dt: float = 0.0
    num_steps: int = 0


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from xyzw to wxyz format (ROS-compatible).

    Parameters
    ----------
    quat_xyzw : np.ndarray
        Quaternion in [x, y, z, w] format. Shape: [..., 4]

    Returns
    -------
    np.ndarray
        Quaternion in [w, x, y, z] format. Shape: [..., 4]
    """
    if quat_xyzw.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with last dimension 4, got shape {quat_xyzw.shape}")
    # Convert [x, y, z, w] -> [w, x, y, z]
    return quat_xyzw[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from wxyz to xyzw format (holosoma internal format).

    Parameters
    ----------
    quat_wxyz : np.ndarray
        Quaternion in [w, x, y, z] format. Shape: [..., 4]

    Returns
    -------
    np.ndarray
        Quaternion in [x, y, z, w] format. Shape: [..., 4]
    """
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with last dimension 4, got shape {quat_wxyz.shape}")
    # Convert [w, x, y, z] -> [x, y, z, w]
    return quat_wxyz[..., [1, 2, 3, 0]]


class StateRecorder:
    """Records proprioceptive and IMU data from MuJoCo simulation.

    This class records data dynamically without a fixed buffer size.
    Saves to pickle format compatible with locomotion_dataset.pkl structure.
    
    Note: Quaternions are stored in wxyz format (ROS-compatible) even though
    holosoma internally uses xyzw format. This conversion happens during save().

    Parameters
    ----------
    simulator : BaseSimulator
        The simulator instance to record from (typically MuJoCo).
    config : StateRecorderConfig, optional
        Configuration for the recorder. If None, uses defaults.

    Example
    -------
    >>> recorder = StateRecorder(simulator)
    >>> for step in range(1000):
    ...     # Apply actions, step simulation...
    ...     recorder.record()
    >>> recorder.save("data.pkl")
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        config: StateRecorderConfig | None = None,
    ):
        self.simulator = simulator
        self.config = config or StateRecorderConfig()

        # Use lists for dynamic growth
        self._timestamps: list[float] = []
        self._dof_pos: list[np.ndarray] = []
        self._dof_vel: list[np.ndarray] = []
        self._torques: list[np.ndarray] = []
        self._base_quat: list[np.ndarray] = []
        self._base_angular_vel: list[np.ndarray] = []
        self._base_linear_acc: list[np.ndarray] = []
        self._root_pos: list[np.ndarray] = []
        self._root_lin_vel: list[np.ndarray] = []

        # Torso IMU data (MuJoCo only)
        self._torso_quat: list[np.ndarray] = []
        self._torso_angular_vel: list[np.ndarray] = []
        self._torso_linear_acc: list[np.ndarray] = []

        # Commands (policy/locomotion commands)
        self._commands: list[np.ndarray] = []

        # Motor commands (low-level motor commands from bridge)
        self._motor_cmd: list[np.ndarray] = []

    def _get_numpy_view(self, tensor, env_id: int = 0) -> np.ndarray:
        """Get numpy copy of tensor data.

        For CPU tensors, copies data. For GPU tensors, copies to CPU first.
        """
        t = tensor[env_id] if tensor.dim() > 1 else tensor
        if t.device.type == "cpu":
            return t.detach().numpy().copy()
        else:
            return t.detach().cpu().numpy().copy()

    def _get_motor_cmd_data(self, env_id: int = 0) -> np.ndarray | None:
        """Extract motor command data from bridge system.

        Parameters
        ----------
        env_id : int
            Environment ID (currently only supports 0 for bridge systems).

        Returns
        -------
        np.ndarray | None
            Motor command data as [num_motors, 5] array where 5 = [q, dq, tau, kp, kd]:
            - q: desired joint position (rad)
            - dq: desired joint velocity (rad/s)
            - tau: feedforward torque (Nm)
            - kp: position gain
            - kd: velocity gain
            Returns None if bridge/motor commands are not available.
        """
        # Check if bridge system is available
        if not hasattr(self.simulator, "bridge") or self.simulator.bridge is None:
            return None

        bridge = self.simulator.bridge
        if not hasattr(bridge, "robot_bridge") or bridge.robot_bridge is None:
            return None

        robot_bridge = bridge.robot_bridge
        if not hasattr(robot_bridge, "low_cmd") or robot_bridge.low_cmd is None:
            return None

        if not hasattr(robot_bridge.low_cmd, "motor_cmd"):
            return None

        motor_cmd_list = robot_bridge.low_cmd.motor_cmd
        if not motor_cmd_list:
            return None

        num_motors = len(motor_cmd_list)
        # Extract motor commands: [q, dq, tau, kp, kd] for each motor
        motor_cmd_array = np.zeros((num_motors, 5), dtype=np.float32)
        for i, motor_cmd in enumerate(motor_cmd_list):
            motor_cmd_array[i, 0] = getattr(motor_cmd, "q", 0.0)
            motor_cmd_array[i, 1] = getattr(motor_cmd, "dq", 0.0)
            motor_cmd_array[i, 2] = getattr(motor_cmd, "tau", 0.0)
            motor_cmd_array[i, 3] = getattr(motor_cmd, "kp", 0.0)
            motor_cmd_array[i, 4] = getattr(motor_cmd, "kd", 0.0)

        return motor_cmd_array

    def record(self, torques: np.ndarray | None = None) -> None:
        """Record current state from simulator.

        Parameters
        ----------
        torques : np.ndarray, optional
            Applied torques to record. If None and record_torques is True,
            will attempt to get torques from simulator (if available).
        """
        env_id = self.config.env_id

        # Record timestamp
        self._timestamps.append(self.simulator.time())

        # Record proprioceptive data
        self._dof_pos.append(self._get_numpy_view(self.simulator.dof_pos, env_id))
        self._dof_vel.append(self._get_numpy_view(self.simulator.dof_vel, env_id))

        # Record torques if available
        if self.config.record_torques:
            if torques is not None:
                self._torques.append(torques.copy())
            elif hasattr(self.simulator, "root_data") and self.simulator.root_data is not None:
                self._torques.append(self.simulator.root_data.ctrl.copy())
            else:
                self._torques.append(np.zeros(self.simulator.num_dof))

        # Record IMU data
        self._base_quat.append(self._get_numpy_view(self.simulator.base_quat, env_id))
        self._base_linear_acc.append(self._get_numpy_view(self.simulator.base_linear_acc, env_id))

        # Record root state (position, linear velocity, angular velocity)
        # robot_root_states format: [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        root_state = self._get_numpy_view(self.simulator.robot_root_states, env_id)
        self._root_pos.append(root_state[0:3].copy())
        self._root_lin_vel.append(root_state[7:10].copy())
        self._base_angular_vel.append(root_state[10:13].copy())  # Angular velocity from root state

        # Record torso IMU if available (MuJoCo only)
        # Check _torso_body_id first - it's set to None if torso body doesn't exist
        torso_body_id = getattr(self.simulator, "_torso_body_id", None)
        if torso_body_id is not None:
            self._torso_quat.append(self._get_numpy_view(self.simulator.torso_quat, env_id))
            self._torso_angular_vel.append(self._get_numpy_view(self.simulator.torso_angular_vel, env_id))
            self._torso_linear_acc.append(self._get_numpy_view(self.simulator.torso_linear_acc, env_id))

        # Record commands if available (locomotion/policy commands)
        if hasattr(self.simulator, "commands") and self.simulator.commands is not None:
            self._commands.append(self._get_numpy_view(self.simulator.commands, env_id))

        # Record motor commands if available (from bridge system)
        motor_cmd_data = self._get_motor_cmd_data(env_id)
        if motor_cmd_data is not None:
            self._motor_cmd.append(motor_cmd_data)

    def get_data(self) -> RecordedData:
        """Get recorded data as a RecordedData object.

        Returns
        -------
        RecordedData
            Container with all recorded data as numpy arrays.
        """
        n = len(self._timestamps)

        return RecordedData(
            timestamps=np.array(self._timestamps, dtype=np.float64),
            dof_pos=np.array(self._dof_pos, dtype=np.float32) if self._dof_pos else np.array([]),
            dof_vel=np.array(self._dof_vel, dtype=np.float32) if self._dof_vel else np.array([]),
            torques=np.array(self._torques, dtype=np.float32) if self._torques else None,
            base_quat=np.array(self._base_quat, dtype=np.float32) if self._base_quat else np.array([]),
            base_angular_vel=np.array(self._base_angular_vel, dtype=np.float32) if self._base_angular_vel else np.array([]),
            base_linear_acc=np.array(self._base_linear_acc, dtype=np.float32) if self._base_linear_acc else np.array([]),
            root_pos=np.array(self._root_pos, dtype=np.float32) if self._root_pos else np.array([]),
            root_lin_vel=np.array(self._root_lin_vel, dtype=np.float32) if self._root_lin_vel else np.array([]),
            torso_quat=np.array(self._torso_quat, dtype=np.float32) if self._torso_quat else None,
            torso_angular_vel=np.array(self._torso_angular_vel, dtype=np.float32) if self._torso_angular_vel else None,
            torso_linear_acc=np.array(self._torso_linear_acc, dtype=np.float32) if self._torso_linear_acc else None,
            commands=np.array(self._commands, dtype=np.float32) if self._commands else None,
            motor_cmd=np.array(self._motor_cmd, dtype=np.float32) if self._motor_cmd else None,
            dof_names=list(self.simulator.dof_names) if hasattr(self.simulator, "dof_names") else [],
            sim_dt=self.simulator.sim_dt,
            num_steps=n,
        )

    def save(self, filepath: str | Path) -> Path:
        """Save recorded data to pickle file.

        Output format is compatible with locomotion_dataset.pkl structure.
        
        Note: Quaternions are automatically converted from xyzw (holosoma internal format)
        to wxyz (ROS-compatible format) when saving to file.

        Saved Data Structure:
        --------------------
        {
            "lowstate_data": {
                "timestamp": [num_steps] (float64)
                "motor_state": {
                    "q": [num_steps, num_dof] (float64) - joint positions
                    "dq": [num_steps, num_dof] (float64) - joint velocities
                    "tau_est": [num_steps, num_dof] (float64) - torques
                    ...
                },
                "imu": {
                    "quaternion": [num_steps, 4] (float32) - wxyz format (ROS)
                    "gyroscope": [num_steps, 3] (float32)
                    "accelerometer": [num_steps, 3] (float32)
                }
            },
            "commands": [num_steps, command_dim] (float32) - locomotion commands (optional)
            "motor_cmd": [num_steps, num_motors, 5] (float32) - motor commands [q, dq, tau, kp, kd] (optional)
            "torso_imu": {
                "quaternion": [num_steps, 4] (float32) - wxyz format (ROS, optional)
                "gyroscope": [num_steps, 3] (float32)
                "accelerometer": [num_steps, 3] (float32)
            } (optional, MuJoCo only)
            "metadata": {...}
        }

        Parameters
        ----------
        filepath : str or Path
            Output file path. Extension .pkl will be added if not present.

        Returns
        -------
        Path
            Path to saved file.
        """
        filepath = Path(filepath)
        if filepath.suffix != ".pkl":
            filepath = filepath.with_suffix(".pkl")

        data = self.get_data()
        num_dof = data.dof_pos.shape[1] if len(data.dof_pos) > 0 else 0

        # Build pickle structure matching locomotion_dataset.pkl
        pickle_data = {
            "sequence_name": self.config.sequence_name,
            "extraction_timestamp": datetime.now().isoformat(),
            "source": "mujoco_simulation",
            # Lowstate data (motor + IMU combined)
            "lowstate_data": {
                "timestamp": data.timestamps.astype(np.float64),  # [num_steps]
                "motor_state": {
                    "mode": np.zeros((data.num_steps, num_dof), dtype=np.float64),  # [num_steps, num_dof]
                    "q": data.dof_pos.astype(np.float64),  # [num_steps, num_dof] - joint positions (rad)
                    "dq": data.dof_vel.astype(np.float64),  # [num_steps, num_dof] - joint velocities (rad/s)
                    "ddq": np.zeros((data.num_steps, num_dof), dtype=np.float64),  # [num_steps, num_dof]
                    "tau_est": data.torques.astype(np.float64) if data.torques is not None else np.zeros((data.num_steps, num_dof), dtype=np.float64),  # [num_steps, num_dof] - torques (Nm)
                    "temperature": np.zeros((data.num_steps, num_dof, 2), dtype=np.float64),  # [num_steps, num_dof, 2]
                },
                "imu": {
                    # Base/pelvis IMU data (wxyz format for ROS compatibility)
                    "quaternion": _xyzw_to_wxyz(data.base_quat).astype(np.float32),  # [num_steps, 4] wxyz (ROS format)
                    "gyroscope": data.base_angular_vel.astype(np.float32),  # [num_steps, 3]
                    "accelerometer": data.base_linear_acc.astype(np.float32),  # [num_steps, 3]
                },
            },
            # Additional metadata
            "metadata": {
                "sim_dt": data.sim_dt,  # Simulation timestep (seconds)
                "num_steps": data.num_steps,  # Number of recorded steps
                "num_dof": num_dof,  # Number of degrees of freedom
                "dof_names": data.dof_names,  # List of joint names
                "root_pos": data.root_pos,  # [num_steps, 3] - root body position (m)
                "root_lin_vel": data.root_lin_vel,  # [num_steps, 3] - root body linear velocity (m/s)
            },
        }

        # Add commands if available (top-level, simulation-specific)
        # Shape: [num_steps, command_dim] - locomotion commands (vx, vy, yaw_rate, walk_stand, etc.)
        if data.commands is not None:
            pickle_data["commands"] = data.commands.astype(np.float32)

        # Add motor commands if available (top-level, simulation-specific)
        # Shape: [num_steps, num_motors, 5] where 5 = [q, dq, tau, kp, kd]
        #   q: desired joint position (rad)
        #   dq: desired joint velocity (rad/s)
        #   tau: feedforward torque (Nm)
        #   kp: position gain
        #   kd: velocity gain
        if data.motor_cmd is not None:
            pickle_data["motor_cmd"] = data.motor_cmd.astype(np.float32)

        # Add torso IMU data at top level if available (simulation-specific, MuJoCo only)
        # Shape: [num_steps, 4] for quaternion, [num_steps, 3] for gyroscope/accelerometer
        if data.torso_quat is not None:
            pickle_data["torso_imu"] = {
                "quaternion": _xyzw_to_wxyz(data.torso_quat).astype(np.float32),  # [num_steps, 4] wxyz (ROS format)
                "gyroscope": data.torso_angular_vel.astype(np.float32) if data.torso_angular_vel is not None else None,  # [num_steps, 3]
                "accelerometer": data.torso_linear_acc.astype(np.float32) if data.torso_linear_acc is not None else None,  # [num_steps, 3]
            }

        with open(filepath, "wb") as f:
            pickle.dump(pickle_data, f)

        duration = data.timestamps[-1] - data.timestamps[0] if data.num_steps > 1 else 0
        sample_rate = data.num_steps / duration if duration > 0 else 0
        logger.info(f"Saved {data.num_steps} steps ({duration:.1f}s @ {sample_rate:.0f}Hz) to {filepath}")

        # Print structure
        self._print_structure(pickle_data, filepath)

        return filepath

    def save_poses(self, filepath: str | Path) -> Path:
        """Save only pelvis poses to a separate pickle file.

        This is a lightweight file containing only the ground truth pose data,
        useful for trajectory analysis or visualization.
        
        Note: Quaternions are automatically converted from xyzw (holosoma internal format)
        to wxyz (ROS-compatible format) when saving to file.

        Parameters
        ----------
        filepath : str or Path
            Output file path. Extension .pkl will be added if not present.

        Returns
        -------
        Path
            Path to saved file.
        """
        filepath = Path(filepath)
        if filepath.suffix != ".pkl":
            filepath = filepath.with_suffix(".pkl")

        data = self.get_data()

        # Lightweight pose-only structure
        # Shapes: [num_steps, ...]
        poses_data = {
            "sequence_name": self.config.sequence_name,
            "timestamp": data.timestamps.astype(np.float64),  # [num_steps]
            "position": data.root_pos.astype(np.float64),  # [num_steps, 3] - root body position in world frame (m)
            "quaternion": _xyzw_to_wxyz(data.base_quat).astype(np.float64),  # [num_steps, 4] - wxyz format (ROS)
            "linear_velocity": data.root_lin_vel.astype(np.float64),  # [num_steps, 3] - root body linear velocity (m/s)
        }

        with open(filepath, "wb") as f:
            pickle.dump(poses_data, f)

        duration = data.timestamps[-1] - data.timestamps[0] if data.num_steps > 1 else 0
        sample_rate = data.num_steps / duration if duration > 0 else 0
        logger.info(f"Saved {data.num_steps} poses ({duration:.1f}s @ {sample_rate:.0f}Hz) to {filepath}")

        # Print structure
        self._print_structure(poses_data, filepath)

        return filepath

    def _print_structure(self, data: dict, filepath: Path, indent: int = 0) -> None:
        """Print the structure and shapes of saved data."""
        prefix = "  " * indent

        if indent == 0:
            print(f"\n{'='*60}")
            print(f"FILE: {filepath}")
            print(f"{'='*60}")

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"{prefix}{key}: shape={value.shape} dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_structure(value, filepath, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}: list[{len(value)}]")
            else:
                val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"{prefix}{key}: {type(value).__name__} = {val_str}")

    @staticmethod
    def load(filepath: str | Path) -> RecordedData:
        """Load recorded data from pickle file.

        Quaternions stored in wxyz format (ROS-compatible) are automatically
        converted back to xyzw format (holosoma internal format) when loading.

        Parameters
        ----------
        filepath : str or Path
            Path to .pkl file.

        Returns
        -------
        RecordedData
            Loaded data container. Quaternions are in xyzw format (holosoma internal).
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        lowstate = data.get("lowstate_data", {})
        motor_state = lowstate.get("motor_state", {})
        imu = lowstate.get("imu", {})
        metadata = data.get("metadata", {})

        # Load torso IMU data if available (top-level, simulation-specific)
        torso_imu = data.get("torso_imu", {})

        # Load commands if available (top-level, simulation-specific)
        commands = data.get("commands")
        motor_cmd = data.get("motor_cmd")

        # Convert quaternions from wxyz (stored format) to xyzw (internal format)
        base_quat_stored = imu.get("quaternion", np.array([]))
        base_quat = _wxyz_to_xyzw(base_quat_stored) if len(base_quat_stored) > 0 else base_quat_stored
        
        torso_quat_stored = torso_imu.get("quaternion") if torso_imu else None
        torso_quat = _wxyz_to_xyzw(torso_quat_stored) if torso_quat_stored is not None and len(torso_quat_stored) > 0 else torso_quat_stored

        return RecordedData(
            timestamps=lowstate.get("timestamp", np.array([])),
            dof_pos=motor_state.get("q", np.array([])),
            dof_vel=motor_state.get("dq", np.array([])),
            torques=motor_state.get("tau_est"),
            base_quat=base_quat,
            base_angular_vel=imu.get("gyroscope", np.array([])),
            base_linear_acc=imu.get("accelerometer", np.array([])),
            root_pos=metadata.get("root_pos", np.array([])),
            root_lin_vel=metadata.get("root_lin_vel", np.array([])),
            torso_quat=torso_quat,
            torso_angular_vel=torso_imu.get("gyroscope") if torso_imu else None,
            torso_linear_acc=torso_imu.get("accelerometer") if torso_imu else None,
            commands=commands,
            motor_cmd=motor_cmd,
            dof_names=metadata.get("dof_names", []),
            sim_dt=metadata.get("sim_dt", 0.0),
            num_steps=metadata.get("num_steps", len(lowstate.get("timestamp", []))),
        )

    def reset(self) -> None:
        """Reset recorder to start fresh recording."""
        self._timestamps.clear()
        self._dof_pos.clear()
        self._dof_vel.clear()
        self._torques.clear()
        self._base_quat.clear()
        self._base_angular_vel.clear()
        self._base_linear_acc.clear()
        self._root_pos.clear()
        self._root_lin_vel.clear()
        self._torso_quat.clear()
        self._torso_angular_vel.clear()
        self._torso_linear_acc.clear()
        self._commands.clear()
        self._motor_cmd.clear()
        logger.debug("StateRecorder reset")

    @property
    def step_count(self) -> int:
        """Number of steps recorded so far."""
        return len(self._timestamps)
