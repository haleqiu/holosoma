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
    """

    # Timestamps
    timestamps: np.ndarray  # [num_steps]

    # Proprioceptive data
    dof_pos: np.ndarray  # [num_steps, num_dof]
    dof_vel: np.ndarray  # [num_steps, num_dof]
    torques: np.ndarray | None  # [num_steps, num_dof] or None

    # IMU data
    base_quat: np.ndarray  # [num_steps, 4] (xyzw format)
    base_angular_vel: np.ndarray  # [num_steps, 3]
    base_linear_acc: np.ndarray  # [num_steps, 3]

    # Root state (position + orientation + velocities)
    root_pos: np.ndarray  # [num_steps, 3]
    root_lin_vel: np.ndarray  # [num_steps, 3]

    # Metadata
    dof_names: list[str] = field(default_factory=list)
    sim_dt: float = 0.0
    num_steps: int = 0


class StateRecorder:
    """Records proprioceptive and IMU data from MuJoCo simulation.

    This class records data dynamically without a fixed buffer size.
    Saves to pickle format compatible with locomotion_dataset.pkl structure.

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

    def _get_numpy_view(self, tensor, env_id: int = 0) -> np.ndarray:
        """Get numpy copy of tensor data.

        For CPU tensors, copies data. For GPU tensors, copies to CPU first.
        """
        t = tensor[env_id] if tensor.dim() > 1 else tensor
        if t.device.type == "cpu":
            return t.detach().numpy().copy()
        else:
            return t.detach().cpu().numpy().copy()

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
            dof_names=list(self.simulator.dof_names) if hasattr(self.simulator, "dof_names") else [],
            sim_dt=self.simulator.sim_dt,
            num_steps=n,
        )

    def save(self, filepath: str | Path) -> Path:
        """Save recorded data to pickle file.

        Output format is compatible with locomotion_dataset.pkl structure.

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
                "timestamp": data.timestamps.astype(np.float64),
                "motor_state": {
                    "mode": np.zeros((data.num_steps, num_dof), dtype=np.float64),
                    "q": data.dof_pos.astype(np.float64),
                    "dq": data.dof_vel.astype(np.float64),
                    "ddq": np.zeros((data.num_steps, num_dof), dtype=np.float64),
                    "tau_est": data.torques.astype(np.float64) if data.torques is not None else np.zeros((data.num_steps, num_dof), dtype=np.float64),
                    "temperature": np.zeros((data.num_steps, num_dof, 2), dtype=np.float64),
                },
                "imu": {
                    "quaternion": data.base_quat.astype(np.float32),
                    "gyroscope": data.base_angular_vel.astype(np.float32),
                    "accelerometer": data.base_linear_acc.astype(np.float32),
                },
            },
            # Additional metadata
            "metadata": {
                "sim_dt": data.sim_dt,
                "num_steps": data.num_steps,
                "num_dof": num_dof,
                "dof_names": data.dof_names,
                "root_pos": data.root_pos,
                "root_lin_vel": data.root_lin_vel,
            },
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
        poses_data = {
            "sequence_name": self.config.sequence_name,
            "timestamp": data.timestamps.astype(np.float64),
            "position": data.root_pos.astype(np.float64),  # [N, 3] world frame
            "quaternion": data.base_quat.astype(np.float64),  # [N, 4] xyzw
            "linear_velocity": data.root_lin_vel.astype(np.float64),  # [N, 3]
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

        Parameters
        ----------
        filepath : str or Path
            Path to .pkl file.

        Returns
        -------
        RecordedData
            Loaded data container.
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        lowstate = data.get("lowstate_data", {})
        motor_state = lowstate.get("motor_state", {})
        imu = lowstate.get("imu", {})
        metadata = data.get("metadata", {})

        return RecordedData(
            timestamps=lowstate.get("timestamp", np.array([])),
            dof_pos=motor_state.get("q", np.array([])),
            dof_vel=motor_state.get("dq", np.array([])),
            torques=motor_state.get("tau_est"),
            base_quat=imu.get("quaternion", np.array([])),
            base_angular_vel=imu.get("gyroscope", np.array([])),
            base_linear_acc=imu.get("accelerometer", np.array([])),
            root_pos=metadata.get("root_pos", np.array([])),
            root_lin_vel=metadata.get("root_lin_vel", np.array([])),
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
        logger.debug("StateRecorder reset")

    @property
    def step_count(self) -> int:
        """Number of steps recorded so far."""
        return len(self._timestamps)
