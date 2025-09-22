import time
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import statistics

@dataclass
class PIDParameters:
    """PID controller parameters"""
    kp: float = 1.0        # Proportional gain
    ki: float = 0.1        # Integral gain
    kd: float = 0.05       # Derivative gain
    setpoint: float = 25.0 # Target temperature
    output_min: float = 0.0 # Minimum output (drive speed %)
    output_max: float = 100.0 # Maximum output (drive speed %)
    integral_windup_limit: float = 100.0 # Anti-windup limit

@dataclass
class ControlLoopConfig:
    """Configuration for a single control loop"""
    loop_id: str
    name: str
    drive_ids: List[int]           # VLT drive IDs (e.g., [1, 2])
    pid_params: PIDParameters
    enabled: bool = False

    # Target configuration - either temperature or flow control
    control_type: str = "temperature"  # "temperature" or "flow"
    temperature_probes: List[str] = None  # Temperature probe tags (e.g., ['T01', 'T02'])
    flow_meters: List[str] = None         # Flow meter tags (e.g., ['F01', 'F02'])

    # Safety limits
    safety_temp_max: float = 100.0  # Emergency stop temperature (for temp control)
    safety_temp_min: float = -10.0  # Emergency stop temperature (for temp control)
    safety_flow_max: float = 100.0  # Emergency stop flow percentage (for flow control)
    safety_flow_min: float = 0.0    # Emergency stop flow percentage (for flow control)

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.control_type not in ["temperature", "flow"]:
            raise ValueError(f"Invalid control_type: {self.control_type}. Must be 'temperature' or 'flow'")

        if self.control_type == "temperature":
            if not self.temperature_probes or len(self.temperature_probes) == 0:
                raise ValueError("Temperature control requires at least one temperature probe")
            if self.flow_meters is not None and len(self.flow_meters) > 0:
                raise ValueError("Temperature control should not have flow_meters configured")
        elif self.control_type == "flow":
            if not self.flow_meters or len(self.flow_meters) == 0:
                raise ValueError("Flow control requires at least one flow meter")
            if self.temperature_probes is not None and len(self.temperature_probes) > 0:
                raise ValueError("Flow control should not have temperature_probes configured")

        if not self.drive_ids or len(self.drive_ids) == 0:
            raise ValueError("Control loop requires at least one drive ID")

        # Validate safety limits
        if self.control_type == "temperature":
            if self.safety_temp_max <= self.safety_temp_min:
                raise ValueError("safety_temp_max must be greater than safety_temp_min")
        else:
            if self.safety_flow_max <= self.safety_flow_min:
                raise ValueError("safety_flow_max must be greater than safety_flow_min")
            if self.safety_flow_min < 0 or self.safety_flow_max > 100:
                raise ValueError("Flow safety limits must be between 0 and 100 percent")

class PIDController:
    """Single PID controller implementation"""

    def __init__(self, params: PIDParameters):
        self.params = params
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.last_input = 0.0
        self.output = 0.0
        self.error_history = []
        self.logger = logging.getLogger(f"{__name__}.PID")

    def update(self, current_value: float, setpoint: Optional[float] = None) -> float:
        """Update PID controller and return output"""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0.0:
            return self.output

        # Update setpoint if provided
        if setpoint is not None:
            self.params.setpoint = setpoint

        # Calculate error
        error = self.params.setpoint - current_value

        # Store error history for analysis (keep last 100 points)
        self.error_history.append((current_time, error))
        if len(self.error_history) > 100:
            self.error_history.pop(0)

        # Proportional term
        proportional = self.params.kp * error

        # Integral term with windup protection
        self.integral += error * dt
        if abs(self.integral) > self.params.integral_windup_limit:
            self.integral = self.params.integral_windup_limit * (1 if self.integral > 0 else -1)
        integral = self.params.ki * self.integral

        # Derivative term (use input derivative to avoid derivative kick)
        input_derivative = (current_value - self.last_input) / dt if dt > 0 else 0
        derivative = -self.params.kd * input_derivative

        # Calculate output
        self.output = proportional + integral + derivative

        # Clamp output to limits
        self.output = max(self.params.output_min, min(self.params.output_max, self.output))

        # Store values for next iteration
        self.last_error = error
        self.last_input = current_value
        self.last_time = current_time

        return self.output

    def reset(self):
        """Reset PID controller state"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.output = 0.0
        self.error_history.clear()

    def get_tuning_info(self) -> Dict[str, float]:
        """Get current tuning information"""
        return {
            'kp': self.params.kp,
            'ki': self.params.ki,
            'kd': self.params.kd,
            'setpoint': self.params.setpoint,
            'output': self.output,
            'integral': self.integral,
            'last_error': self.last_error
        }

class ControlLoop:
    """Single temperature control loop"""

    def __init__(self, config: ControlLoopConfig):
        self.config = config
        self.pid = PIDController(config.pid_params)
        self.enabled = config.enabled
        self.last_temperature = None
        self.last_output = 0.0
        self.safety_triggered = False
        self.loop_start_time = None
        self.logger = logging.getLogger(f"{__name__}.Loop.{config.loop_id}")

    def update(self, sensor_data: Dict[str, float], timestamp: datetime) -> Tuple[float, Dict[str, Any]]:
        """Update control loop and return average drive speed and status info"""

        if self.config.control_type == "temperature":
            return self._update_temperature_control(sensor_data, timestamp)
        elif self.config.control_type == "flow":
            return self._update_flow_control(sensor_data, timestamp)
        else:
            self.logger.error(f"Unknown control type: {self.config.control_type}")
            return 0.0, self._get_status_info(timestamp, None, "Configuration Error")

    def _update_temperature_control(self, sensor_data: Dict[str, float], timestamp: datetime) -> Tuple[float, Dict[str, Any]]:
        """Update temperature control loop"""
        # Calculate average temperature from assigned probes
        temperatures = []
        if self.config.temperature_probes:
            for probe in self.config.temperature_probes:
                if probe in sensor_data and sensor_data[probe] is not None:
                    temp_value = sensor_data[probe]
                    if not (temp_value != temp_value):  # Check for NaN
                        temperatures.append(temp_value)

        if not temperatures:
            self.logger.warning(f"No valid temperature readings for loop {self.config.loop_id}")
            return self.last_output, self._get_status_info(timestamp, None, "No Temperature Data")

        avg_temperature = statistics.mean(temperatures)
        self.last_temperature = avg_temperature

        # Safety checks
        if (avg_temperature > self.config.safety_temp_max or
            avg_temperature < self.config.safety_temp_min):
            self.safety_triggered = True
            self.enabled = False
            self.logger.error(f"Temperature safety limit triggered in loop {self.config.loop_id}: {avg_temperature}°C")
            return 0.0, self._get_status_info(timestamp, avg_temperature, "Safety Stop")

        # If disabled, return zero output
        if not self.enabled:
            return 0.0, self._get_status_info(timestamp, avg_temperature, "Disabled")

        # Update PID controller
        output = self.pid.update(avg_temperature)
        self.last_output = output

        status = "Active" if self.enabled else "Disabled"
        return output, self._get_status_info(timestamp, avg_temperature, status)

    def _update_flow_control(self, sensor_data: Dict[str, float], timestamp: datetime) -> Tuple[float, Dict[str, Any]]:
        """Update flow control loop"""
        # Calculate average flow from assigned flow meters
        flows = []
        if self.config.flow_meters:
            for meter in self.config.flow_meters:
                if meter in sensor_data and sensor_data[meter] is not None:
                    flow_value = sensor_data[meter]
                    if not (flow_value != flow_value):  # Check for NaN
                        flows.append(flow_value)

        if not flows:
            self.logger.warning(f"No valid flow readings for loop {self.config.loop_id}")
            return self.last_output, self._get_status_info(timestamp, None, "No Flow Data")

        avg_flow = statistics.mean(flows)
        self.last_temperature = avg_flow  # Reuse this field for flow value

        # Safety checks
        if (avg_flow > self.config.safety_flow_max or
            avg_flow < self.config.safety_flow_min):
            self.safety_triggered = True
            self.enabled = False
            self.logger.error(f"Flow safety limit triggered in loop {self.config.loop_id}: {avg_flow}%")
            return 0.0, self._get_status_info(timestamp, avg_flow, "Safety Stop")

        # If disabled, return zero output
        if not self.enabled:
            return 0.0, self._get_status_info(timestamp, avg_flow, "Disabled")

        # Update PID controller
        output = self.pid.update(avg_flow)
        self.last_output = output

        status = "Active" if self.enabled else "Disabled"
        return output, self._get_status_info(timestamp, avg_flow, status)

    def _get_status_info(self, timestamp: datetime, temperature: Optional[float], status: str) -> Dict[str, Any]:
        """Get comprehensive status information"""
        return {
            'loop_id': self.config.loop_id,
            'name': self.config.name,
            'timestamp': timestamp,
            'enabled': self.enabled,
            'status': status,
            'temperature': temperature,
            'setpoint': self.config.pid_params.setpoint,
            'output': self.last_output,
            'error': self.config.pid_params.setpoint - temperature if temperature is not None else None,
            'safety_triggered': self.safety_triggered,
            'pid_info': self.pid.get_tuning_info(),
            'probe_count': len(self.config.temperature_probes),
            'drive_count': len(self.config.drive_ids)
        }

    def enable(self):
        """Enable the control loop"""
        if self.safety_triggered:
            self.logger.warning(f"Cannot enable loop {self.config.loop_id}: safety triggered")
            return False

        self.enabled = True
        self.loop_start_time = datetime.now()
        self.pid.reset()  # Reset PID state when enabling
        self.logger.info(f"Enabled control loop {self.config.loop_id}")
        return True

    def disable(self):
        """Disable the control loop"""
        self.enabled = False
        self.loop_start_time = None
        self.logger.info(f"Disabled control loop {self.config.loop_id}")

    def reset_safety(self):
        """Reset safety trigger (allows re-enabling)"""
        self.safety_triggered = False
        self.logger.info(f"Reset safety trigger for loop {self.config.loop_id}")

    def update_setpoint(self, setpoint: float):
        """Update target setpoint (temperature or flow)"""
        self.config.pid_params.setpoint = setpoint
        unit = "°C" if self.config.control_type == "temperature" else "%"
        self.logger.info(f"Updated setpoint for loop {self.config.loop_id} to {setpoint}{unit}")

    def update_pid_params(self, kp: float = None, ki: float = None, kd: float = None):
        """Update PID tuning parameters"""
        if kp is not None:
            self.config.pid_params.kp = kp
        if ki is not None:
            self.config.pid_params.ki = ki
        if kd is not None:
            self.config.pid_params.kd = kd

        # Update the PID controller parameters
        self.pid.params = self.config.pid_params
        self.logger.info(f"Updated PID parameters for loop {self.config.loop_id}")

class AutoControlSystem:
    """Main auto control system managing multiple PID loops"""

    def __init__(self):
        self.control_loops: Dict[str, ControlLoop] = {}
        self.enabled = False
        self.update_frequency = 1.0  # seconds
        self.control_task = None
        self.last_update_time = None
        self.system_start_time = None
        self.logger = logging.getLogger(f"{__name__}.AutoControl")

        # System-wide safety
        self.emergency_stop_active = False
        self.max_drive_speed = 100.0
        self.min_drive_speed = 0.0

    def add_control_loop(self, config: ControlLoopConfig) -> bool:
        """Add a new control loop"""
        try:
            if config.loop_id in self.control_loops:
                self.logger.warning(f"Control loop {config.loop_id} already exists")
                return False

            loop = ControlLoop(config)
            self.control_loops[config.loop_id] = loop
            self.logger.info(f"Added control loop: {config.loop_id} ({config.name})")
            return True

        except Exception as e:
            self.logger.error(f"Error adding control loop {config.loop_id}: {e}")
            return False

    def remove_control_loop(self, loop_id: str) -> bool:
        """Remove a control loop"""
        if loop_id in self.control_loops:
            self.control_loops[loop_id].disable()
            del self.control_loops[loop_id]
            self.logger.info(f"Removed control loop: {loop_id}")
            return True
        return False

    def get_control_loop(self, loop_id: str) -> Optional[ControlLoop]:
        """Get a specific control loop"""
        return self.control_loops.get(loop_id)

    def get_all_loops(self) -> Dict[str, ControlLoop]:
        """Get all control loops"""
        return self.control_loops.copy()

    async def start_control_system(self) -> bool:
        """Start the auto control system"""
        try:
            if self.enabled:
                self.logger.warning("Control system already running")
                return True  # Already running is considered success

            self.enabled = True
            self.system_start_time = datetime.now()
            self.emergency_stop_active = False

            # Start control task
            self.control_task = asyncio.create_task(self._control_loop())

            self.logger.info("Auto control system started")
            return True

        except Exception as e:
            self.logger.error(f"Error starting control system: {e}")
            return False

    async def stop_control_system(self) -> bool:
        """Stop the auto control system"""
        try:
            self.enabled = False

            # Disable all control loops
            for loop in self.control_loops.values():
                loop.disable()

            # Cancel control task
            if self.control_task:
                self.control_task.cancel()
                try:
                    await self.control_task
                except asyncio.CancelledError:
                    pass

            self.system_start_time = None
            self.logger.info("Auto control system stopped")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping control system: {e}")
            return False

    async def emergency_stop(self) -> bool:
        """Emergency stop all control loops"""
        try:
            self.emergency_stop_active = True

            # Disable all loops immediately
            for loop in self.control_loops.values():
                loop.disable()

            self.logger.warning("Emergency stop activated - all control loops disabled")
            return True

        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
            return False

    async def _control_loop(self):
        """Main control loop that runs continuously"""
        while self.enabled:
            try:
                self.last_update_time = datetime.now()

                # This will be called by the main application with current data
                # The actual control logic is in update_control_loops()

                await asyncio.sleep(self.update_frequency)

            except asyncio.CancelledError:
                self.logger.info("Control loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in control loop: {e}")
                await asyncio.sleep(1.0)

    async def update_control_loops(self, temperature_data: Dict[str, float],
                                  vlt_controller, timestamp: datetime = None) -> Dict[str, Any]:
        """Update all control loops and apply outputs to VLT drives"""
        if not self.enabled or self.emergency_stop_active:
            return {}

        if timestamp is None:
            timestamp = datetime.now()

        results = {}

        try:
            for loop_id, loop in self.control_loops.items():
                if not loop.enabled:
                    continue

                # Update the control loop
                drive_speed, status_info = loop.update(temperature_data, timestamp)
                results[loop_id] = status_info

                # Apply speed to assigned drives
                if vlt_controller and vlt_controller.is_connected():
                    for drive_id in loop.config.drive_ids:
                        try:
                            # Clamp speed to system-wide limits
                            safe_speed = max(self.min_drive_speed,
                                           min(self.max_drive_speed, drive_speed))

                            await vlt_controller.set_drive_speed(drive_id, safe_speed)

                        except Exception as e:
                            self.logger.error(f"Error setting drive {drive_id} speed: {e}")

        except Exception as e:
            self.logger.error(f"Error updating control loops: {e}")

        return results

    def is_running(self) -> bool:
        """Check if the control system is running"""
        return self.enabled

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        active_loops = sum(1 for loop in self.control_loops.values() if loop.enabled)
        safety_triggered = any(loop.safety_triggered for loop in self.control_loops.values())

        return {
            'enabled': self.enabled,
            'emergency_stop': self.emergency_stop_active,
            'total_loops': len(self.control_loops),
            'active_loops': active_loops,
            'safety_triggered': safety_triggered,
            'update_frequency': self.update_frequency,
            'system_start_time': self.system_start_time,
            'last_update_time': self.last_update_time
        }

    def set_update_frequency(self, frequency: float):
        """Set the control system update frequency"""
        self.update_frequency = max(0.1, min(10.0, frequency))
        self.logger.info(f"Updated control frequency to {self.update_frequency}s")

    def set_drive_speed_limits(self, min_speed: float, max_speed: float):
        """Set system-wide drive speed limits"""
        self.min_drive_speed = max(0.0, min_speed)
        self.max_drive_speed = min(100.0, max_speed)
        self.logger.info(f"Updated drive speed limits: {self.min_drive_speed}% - {self.max_drive_speed}%")