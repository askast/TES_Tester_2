import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import plotly.graph_objects as go
from nicegui import ui, app, background_tasks
from daq970a import DAQ970A
from vlt_drives import VLTDriveController
from adam4024 import ADAM4024Controller
from config import TEMPERATURE_CHANNELS, FLOW_CHANNELS, VLT_CONFIG
from database import DataRecorder
from pid_controller import AutoControlSystem, ControlLoopConfig, PIDParameters
import weakref
import time
import serial.tools.list_ports
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Manages saving and loading of complete system configuration"""

    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.ConfigManager")

    def save_configuration(self, hmi_instance, config_name: str) -> bool:
        """Save complete HMI configuration to file"""
        try:
            config_data = self._build_config_data(hmi_instance, config_name)
            config_file = self.config_dir / f"{config_name}.json"

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.logger.info(f"Configuration saved to {config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def _build_config_data(self, hmi_instance, config_name: str) -> Dict[str, Any]:
        """Build configuration data dictionary from HMI instance"""
        return {
            'metadata': {
                'name': config_name,
                'created': datetime.now().isoformat(),
                'version': '1.0'
            },
            'equipment': {
                'daq_ip': hmi_instance.daq_ip,
                'modbus_port': hmi_instance.modbus_port,
                'adam4024_unit_id': hmi_instance.adam4024_unit_id,
                'vlt_unit_ids': hmi_instance.vlt_unit_ids.copy()
            },
            'channels': {
                'temperature': {
                    'count': len(hmi_instance.temp_config['mappings']),
                    'sensor_type': hmi_instance.temp_config['sensor_type'],
                    'mappings': hmi_instance.temp_config['mappings'].copy()
                },
                'flow': {
                    'count': len(hmi_instance.flow_config['mappings']),
                    'range': hmi_instance.flow_config['range'],
                    'min_current': hmi_instance.flow_config['min_current'],
                    'max_current': hmi_instance.flow_config['max_current'],
                    'mappings': hmi_instance.flow_config['mappings'].copy()
                }
            },
            'recording': {
                'recording_frequency': hmi_instance.recording_frequency,
                'update_interval': hmi_instance.update_interval
            },
            'control_loops': self._export_control_loops(hmi_instance.auto_control)
        }

    def load_configuration(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            config_file = self.config_dir / f"{config_name}.json"
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Validate configuration structure
            if not self.validate_configuration(config_data):
                raise ValueError("Invalid configuration file structure")

            self.logger.info(f"Configuration loaded from {config_file}")
            return config_data

        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}

    def apply_configuration(self, hmi_instance, config_data: Dict[str, Any]) -> bool:
        """Apply loaded configuration to HMI instance"""
        try:
            if hmi_instance.is_connected():
                raise ValueError("Cannot apply configuration while equipment is connected")

            # Apply equipment settings
            if 'equipment' in config_data:
                eq = config_data['equipment']
                hmi_instance.daq_ip = eq.get('daq_ip', hmi_instance.daq_ip)
                hmi_instance.modbus_port = eq.get('modbus_port', hmi_instance.modbus_port)
                hmi_instance.adam4024_unit_id = eq.get('adam4024_unit_id', hmi_instance.adam4024_unit_id)
                hmi_instance.vlt_unit_ids = eq.get('vlt_unit_ids', hmi_instance.vlt_unit_ids)

            # Apply channel configurations
            if 'channels' in config_data:
                channels = config_data['channels']

                if 'temperature' in channels:
                    temp_config = channels['temperature']
                    hmi_instance.temp_config = {
                        'sensor_type': temp_config.get('sensor_type', 'K'),
                        'mappings': temp_config.get('mappings', {})
                    }
                    hmi_instance.refresh_temperature_cards()

                if 'flow' in channels:
                    flow_config = channels['flow']
                    hmi_instance.flow_config = {
                        'range': flow_config.get('range', 0.02),
                        'min_current': flow_config.get('min_current', 4),
                        'max_current': flow_config.get('max_current', 20),
                        'mappings': flow_config.get('mappings', {})
                    }
                    hmi_instance.refresh_flow_cards()

            # Apply recording settings
            if 'recording' in config_data:
                rec = config_data['recording']
                hmi_instance.recording_frequency = rec.get('recording_frequency', hmi_instance.recording_frequency)
                hmi_instance.update_interval = rec.get('update_interval', hmi_instance.update_interval)

            # Apply control loops
            if 'control_loops' in config_data:
                self._import_control_loops(hmi_instance.auto_control, config_data['control_loops'])

            self.logger.info("Configuration applied successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error applying configuration: {e}")
            return False

    def list_configurations(self) -> List[Dict[str, str]]:
        """List all available configuration files"""
        configs = []
        try:
            for config_file in self.config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)
                    metadata = data.get('metadata', {})
                    configs.append({
                        'name': config_file.stem,
                        'display_name': metadata.get('name', config_file.stem),
                        'created': metadata.get('created', 'Unknown'),
                        'file_path': str(config_file)
                    })
                except Exception as e:
                    self.logger.warning(f"Could not read config file {config_file}: {e}")
        except Exception as e:
            self.logger.error(f"Error listing configurations: {e}")

        return sorted(configs, key=lambda x: x['created'], reverse=True)

    def delete_configuration(self, config_name: str) -> bool:
        """Delete a configuration file"""
        try:
            config_file = self.config_dir / f"{config_name}.json"
            if config_file.exists():
                config_file.unlink()
                self.logger.info(f"Deleted configuration: {config_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting configuration: {e}")
            return False

    def _export_control_loops(self, auto_control) -> List[Dict[str, Any]]:
        """Export control loop configurations"""
        loops = []
        try:
            for loop_id, loop in auto_control.get_all_loops().items():
                config = loop.config
                loop_data = {
                    'loop_id': config.loop_id,
                    'name': config.name,
                    'control_type': getattr(config, 'control_type', 'temperature'),
                    'temperature_probes': getattr(config, 'temperature_probes', None),
                    'flow_meters': getattr(config, 'flow_meters', None),
                    'drive_ids': config.drive_ids,
                    'enabled': config.enabled,
                    'pid_params': {
                        'kp': config.pid_params.kp,
                        'ki': config.pid_params.ki,
                        'kd': config.pid_params.kd,
                        'setpoint': config.pid_params.setpoint,
                        'output_min': config.pid_params.output_min,
                        'output_max': config.pid_params.output_max,
                        'integral_windup_limit': config.pid_params.integral_windup_limit
                    },
                    'safety_temp_max': config.safety_temp_max,
                    'safety_temp_min': config.safety_temp_min,
                    'safety_flow_max': getattr(config, 'safety_flow_max', 100.0),
                    'safety_flow_min': getattr(config, 'safety_flow_min', 0.0)
                }
                loops.append(loop_data)
        except Exception as e:
            self.logger.error(f"Error exporting control loops: {e}")

        return loops

    def _import_control_loops(self, auto_control, loop_data: List[Dict[str, Any]]) -> bool:
        """Import control loop configurations"""
        try:
            # Clear existing loops
            existing_loops = list(auto_control.get_all_loops().keys())
            for loop_id in existing_loops:
                auto_control.remove_control_loop(loop_id)

            # Import new loops
            for loop_config in loop_data:
                from pid_controller import PIDParameters, ControlLoopConfig

                pid_params = PIDParameters(
                    kp=loop_config['pid_params']['kp'],
                    ki=loop_config['pid_params']['ki'],
                    kd=loop_config['pid_params']['kd'],
                    setpoint=loop_config['pid_params']['setpoint'],
                    output_min=loop_config['pid_params']['output_min'],
                    output_max=loop_config['pid_params']['output_max'],
                    integral_windup_limit=loop_config['pid_params']['integral_windup_limit']
                )

                config = ControlLoopConfig(
                    loop_id=loop_config['loop_id'],
                    name=loop_config['name'],
                    control_type=loop_config.get('control_type', 'temperature'),
                    temperature_probes=loop_config.get('temperature_probes'),
                    flow_meters=loop_config.get('flow_meters'),
                    drive_ids=loop_config['drive_ids'],
                    pid_params=pid_params,
                    enabled=loop_config.get('enabled', False),
                    safety_temp_max=loop_config.get('safety_temp_max', 100.0),
                    safety_temp_min=loop_config.get('safety_temp_min', -10.0),
                    safety_flow_max=loop_config.get('safety_flow_max', 100.0),
                    safety_flow_min=loop_config.get('safety_flow_min', 0.0)
                )

                auto_control.add_control_loop(config)

            return True

        except Exception as e:
            self.logger.error(f"Error importing control loops: {e}")
            return False

    def validate_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Validate configuration file structure and content"""
        try:
            # Check required top-level sections
            required_sections = ['metadata', 'equipment', 'channels', 'recording']
            for section in required_sections:
                if section not in config_data:
                    self.logger.warning(f"Missing required section: {section}")
                    return False

            # Validate metadata
            metadata = config_data['metadata']
            if not isinstance(metadata.get('name'), str) or not metadata['name']:
                self.logger.warning("Invalid or missing metadata.name")
                return False

            # Validate equipment settings
            equipment = config_data['equipment']
            if not isinstance(equipment.get('daq_ip'), str):
                self.logger.warning("Invalid equipment.daq_ip")
                return False

            if not isinstance(equipment.get('vlt_unit_ids'), list):
                self.logger.warning("Invalid equipment.vlt_unit_ids")
                return False

            # Validate channel configurations
            channels = config_data['channels']
            if 'temperature' in channels:
                temp_config = channels['temperature']
                if not isinstance(temp_config.get('count'), int) or temp_config['count'] < 1:
                    self.logger.warning("Invalid temperature channel count")
                    return False

                if not isinstance(temp_config.get('mappings'), dict):
                    self.logger.warning("Invalid temperature mappings")
                    return False

                # Validate temperature channel count matches mappings
                if len(temp_config['mappings']) != temp_config['count']:
                    self.logger.warning("Temperature channel count mismatch")
                    return False

            if 'flow' in channels:
                flow_config = channels['flow']
                if not isinstance(flow_config.get('count'), int) or flow_config['count'] < 1:
                    self.logger.warning("Invalid flow channel count")
                    return False

                if not isinstance(flow_config.get('mappings'), dict):
                    self.logger.warning("Invalid flow mappings")
                    return False

                # Validate flow channel count matches mappings
                if len(flow_config['mappings']) != flow_config['count']:
                    self.logger.warning("Flow channel count mismatch")
                    return False

            # Validate recording settings
            recording = config_data['recording']
            if not isinstance(recording.get('recording_frequency'), (int, float)) or recording['recording_frequency'] <= 0:
                self.logger.warning("Invalid recording frequency")
                return False

            # Validate control loops if present
            if 'control_loops' in config_data:
                loops = config_data['control_loops']
                if not isinstance(loops, list):
                    self.logger.warning("Invalid control loops format")
                    return False

                for loop in loops:
                    if not self._validate_control_loop(loop):
                        return False

            self.logger.info("Configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    def _validate_control_loop(self, loop_data: Dict[str, Any]) -> bool:
        """Validate individual control loop configuration"""
        try:
            required_fields = ['loop_id', 'name', 'drive_ids', 'pid_params']
            for field in required_fields:
                if field not in loop_data:
                    self.logger.warning(f"Missing required control loop field: {field}")
                    return False

            # Validate control type
            control_type = loop_data.get('control_type', 'temperature')
            if control_type not in ['temperature', 'flow']:
                self.logger.warning(f"Invalid control type: {control_type}")
                return False

            # Validate sensors based on control type
            if control_type == 'temperature':
                if not loop_data.get('temperature_probes'):
                    self.logger.warning("Temperature control loop missing temperature_probes")
                    return False
            else:
                if not loop_data.get('flow_meters'):
                    self.logger.warning("Flow control loop missing flow_meters")
                    return False

            # Validate PID parameters
            pid_params = loop_data['pid_params']
            required_pid_fields = ['kp', 'ki', 'kd', 'setpoint']
            for field in required_pid_fields:
                if field not in pid_params or not isinstance(pid_params[field], (int, float)):
                    self.logger.warning(f"Invalid PID parameter: {field}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating control loop: {e}")
            return False

    def export_configuration(self, hmi_instance, export_path: str) -> bool:
        """Export configuration to a specific file path"""
        try:
            config_name = Path(export_path).stem
            config_data = self._build_config_data(hmi_instance, config_name)

            with open(export_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.logger.info(f"Configuration exported to {export_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False

    def import_configuration(self, import_path: str) -> Dict[str, Any]:
        """Import configuration from a specific file path"""
        try:
            if not os.path.exists(import_path):
                raise FileNotFoundError(f"Import file not found: {import_path}")

            with open(import_path, 'r') as f:
                config_data = json.load(f)

            if not self.validate_configuration(config_data):
                raise ValueError("Invalid configuration file")

            self.logger.info(f"Configuration imported from {import_path}")
            return config_data

        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            return {}

class DataCollectionHMI:
    def __init__(self):
        self.daq = None
        self.vlt_controller = None
        self.adam4024_controller = None
        self.data_history = []
        self.update_interval = 1.0  # seconds
        self.is_running = False

        # Configuration
        self.daq_ip = "192.168.1.100"  # Default IP - user can change
        self.modbus_port = "COM1"      # Default port - user can change
        self.adam4024_unit_id = 7      # Default Modbus unit ID for ADAM-4024
        self.vlt_unit_ids = VLT_CONFIG['unit_ids'].copy()  # VLT drive unit IDs

        # Data recording configuration
        self.test_name = ""
        self.test_description = ""
        self.recording_frequency = 1.0  # seconds
        self.data_recorder = DataRecorder()
        self.equipment_locked = False  # Lock parameters when connected and recording

        # Auto control system
        self.auto_control = AutoControlSystem()
        self.control_enabled = False

        # Example control loop configurations:
        #
        # Temperature Control Example:
        # ControlLoopConfig(
        #     loop_id="TEMP_LOOP_1",
        #     name="Main Temperature Control",
        #     control_type="temperature",
        #     temperature_probes=["T01", "T02"],  # Use probes T01 and T02
        #     flow_meters=None,
        #     drive_ids=[1, 2],  # Control VLT drives 1 and 2
        #     pid_params=PIDParameters(kp=1.0, ki=0.1, kd=0.05, setpoint=25.0),
        #     safety_temp_max=100.0,
        #     safety_temp_min=-10.0
        # )
        #
        # Flow Control Example:
        # ControlLoopConfig(
        #     loop_id="FLOW_LOOP_1",
        #     name="Coolant Flow Control",
        #     control_type="flow",
        #     temperature_probes=None,
        #     flow_meters=["F01", "F02"],  # Use flow meters F01 and F02
        #     drive_ids=[3, 4],  # Control VLT drives 3 and 4
        #     pid_params=PIDParameters(kp=2.0, ki=0.2, kd=0.1, setpoint=50.0),  # 50% flow
        #     safety_flow_max=95.0,  # Maximum 95% flow
        #     safety_flow_min=5.0    # Minimum 5% flow
        # )

        # Channel configurations (can be modified in settings)
        self.temp_config = TEMPERATURE_CHANNELS.copy()
        self.flow_config = FLOW_CHANNELS.copy()

        # UI elements (will be set during UI creation)
        self.connection_status = None
        self.temp_cards = {}
        self.flow_cards = {}
        self.drive_cards = {}
        self.valve_cards = {}
        self.temp_chart = None
        self.flow_chart = None
        self.channel_config_dialogs = {}

        # Task management
        self.data_task = None
        self.keepalive_task = None
        self.last_update_time = time.time()

        # UI responsiveness
        self.ui_update_queue = None  # Will be initialized when event loop is ready
        self.ui_responsive = True
        self.background_tasks_started = False

        # Serial port management
        self.available_ports = []
        self.port_select = None

        # Configuration management
        self.config_manager = ConfigurationManager()

    async def connect_equipment(self):
        try:
            # Connect to DAQ970A
            if self.daq_ip:
                self.daq = DAQ970A(self.daq_ip, temp_config=self.temp_config, flow_config=self.flow_config)
                daq_connected = await self.daq.connect()
                if daq_connected:
                    ui.notify("DAQ970A connected successfully", type="positive")
                else:
                    ui.notify("Failed to connect to DAQ970A", type="negative")

            # Connect to VLT drives and ADAM-4024
            selected_port = self.get_selected_port()
            if selected_port:
                self.vlt_controller = VLTDriveController(selected_port, unit_ids=self.vlt_unit_ids)
                vlt_connected = await self.vlt_controller.connect()
                if vlt_connected:
                    ui.notify(f"VLT drives connected successfully on {selected_port}", type="positive")

                    # Connect ADAM-4024 on same Modbus network
                    self.adam4024_controller = ADAM4024Controller(unit_id=self.adam4024_unit_id)
                    self.adam4024_controller.set_modbus_client(self.vlt_controller.client)

                    adam_connected = await self.adam4024_controller.initialize_device()
                    if adam_connected:
                        ui.notify(f"ADAM-4024 connected successfully (Unit ID: {self.adam4024_unit_id})", type="positive")

                        # Configure default valve names
                        self.adam4024_controller.configure_channel(1, "Main Control Valve", "Flow Control")
                        self.adam4024_controller.configure_channel(2, "Bypass Valve", "Flow Control")
                        self.adam4024_controller.configure_channel(3, "Pressure Relief Valve", "Safety")
                        self.adam4024_controller.configure_channel(4, "Isolation Valve", "On/Off Control")
                    else:
                        ui.notify(f"Failed to connect to ADAM-4024 (Unit ID: {self.adam4024_unit_id})", type="warning")
                else:
                    ui.notify(f"Failed to connect to VLT drives on {selected_port}", type="negative")
            else:
                ui.notify("Please select a Modbus port", type="warning")

            # Update connection status
            await self.update_connection_status()

            # Update parameter lock status based on recording state
            if self.data_recorder.is_recording():
                self.equipment_locked = True
            self.update_parameter_lock_status()

        except Exception as e:
            logger.error(f"Error connecting equipment: {e}")
            ui.notify(f"Connection error: {e}", type="negative")

    async def disconnect_equipment(self):
        try:
            # Stop data collection if running
            if self.is_running:
                await self.toggle_logging()

            if self.daq:
                await self.daq.disconnect()
                self.daq = None

            if self.vlt_controller:
                await self.vlt_controller.disconnect()
                self.vlt_controller = None

            if self.adam4024_controller:
                # Emergency stop all valves before disconnect
                await self.adam4024_controller.emergency_stop()
                self.adam4024_controller = None

            await self.update_connection_status()

            # Unlock parameters when disconnected (unless recording)
            if not self.data_recorder.is_recording():
                self.equipment_locked = False
                self.update_parameter_lock_status()

            ui.notify("Equipment disconnected", type="info")

        except Exception as e:
            logger.error(f"Error disconnecting equipment: {e}")

    async def cleanup_on_disconnect(self):
        """Cleanup when client disconnects"""
        try:
            logger.info("Client disconnected, cleaning up resources")

            # Stop all running tasks
            self.is_running = False

            # Cancel background tasks
            if self.data_task and not self.data_task.done():
                self.data_task.cancel()

            if hasattr(self, 'ui_processor_task') and not self.ui_processor_task.done():
                self.ui_processor_task.cancel()

            if self.keepalive_task and not self.keepalive_task.done():
                self.keepalive_task.cancel()

            # Disconnect equipment
            if self.daq:
                await self.daq.disconnect()

            if self.vlt_controller:
                await self.vlt_controller.disconnect()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def start_background_tasks(self):
        """Initialize background tasks when event loop is ready"""
        try:
            if not self.background_tasks_started:
                # Initialize the UI update queue
                self.ui_update_queue = asyncio.Queue()

                # Start keepalive task
                self.keepalive_task = background_tasks.create(self.keepalive_task_func())

                self.background_tasks_started = True
                logger.info("Background tasks started successfully")

        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")

    def get_available_serial_ports(self) -> List[Dict[str, str]]:
        """Get list of available serial ports"""
        try:
            ports = serial.tools.list_ports.comports()
            port_list = []

            for port in ports:
                port_info = {
                    'device': port.device,
                    'name': port.name or port.device,
                    'description': port.description or 'No description',
                    'hwid': port.hwid or 'Unknown',
                    'manufacturer': getattr(port, 'manufacturer', 'Unknown'),
                    'display_name': f"{port.device} - {port.description or 'Serial Port'}"
                }
                port_list.append(port_info)

            # Sort by device name
            port_list.sort(key=lambda x: x['device'])

            # Add common RS485 Modbus ports if not already detected
            common_rs485_ports = [
                '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2',
                '/dev/ttyCH340USB0', '/dev/ttyCH340USB1', '/dev/ttyCH340USB2',
                '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2',
                '/dev/ttyRS4850', '/dev/ttyRS4851',
                'COM1', 'COM2', 'COM3', 'COM4', 'COM5'
            ]

            existing_devices = [port['device'] for port in port_list]

            for common_port in common_rs485_ports:
                if common_port not in existing_devices:
                    # Determine port type and description
                    if 'ttyACM' in common_port:
                        description = 'RS485 Modbus Port (ACM)'
                    elif 'ttyCH340USB' in common_port:
                        description = 'RS485 Modbus Port (CH340 USB)'
                    elif 'ttyUSB' in common_port:
                        description = 'RS485 Modbus Port (USB)'
                    elif 'ttyRS485' in common_port:
                        description = 'RS485 Modbus Port'
                    elif 'COM' in common_port:
                        description = 'RS485 Modbus Port (Windows)'
                    else:
                        description = 'RS485 Modbus Port'

                    port_list.append({
                        'device': common_port,
                        'name': common_port,
                        'description': description,
                        'hwid': 'Common RS485 Port',
                        'manufacturer': 'Common',
                        'display_name': f"{common_port} - {description}"
                    })

            # Sort again after adding common ports
            detected_ports = [p for p in port_list if p.get('manufacturer') != 'Common']
            common_ports = [p for p in port_list if p.get('manufacturer') == 'Common']

            port_list = detected_ports + common_ports

            # Add manual entry option at the top
            port_list.insert(0, {
                'device': 'manual',
                'name': 'Manual Entry',
                'description': 'Enter port manually',
                'hwid': '',
                'manufacturer': '',
                'display_name': 'Manual Entry'
            })

            return port_list

        except Exception as e:
            logger.error(f"Error getting serial ports: {e}")
            # Return common RS485 ports as fallback
            return [
                {'device': 'manual', 'display_name': 'Manual Entry', 'description': 'Enter port manually'},
                {'device': 'COM1', 'display_name': 'COM1 - RS485 Modbus Port (Windows)', 'description': 'Windows RS485 port'},
                {'device': '/dev/ttyACM1', 'display_name': '/dev/ttyACM1 - RS485 Modbus Port (ACM)', 'description': 'Linux RS485 ACM port'},
                {'device': '/dev/ttyCH340USB1', 'display_name': '/dev/ttyCH340USB1 - RS485 Modbus Port (CH340)', 'description': 'Linux RS485 CH340 USB port'},
                {'device': '/dev/ttyUSB0', 'display_name': '/dev/ttyUSB0 - RS485 Modbus Port (USB)', 'description': 'Linux RS485 USB port'}
            ]

    def refresh_serial_ports(self):
        """Refresh the list of available serial ports"""
        try:
            self.available_ports = self.get_available_serial_ports()

            if self.port_select:
                # Update the select options
                options = [port['display_name'] for port in self.available_ports]
                values = [port['device'] for port in self.available_ports]

                # Store current selection
                current_selection = self.port_select.value

                # Update options
                self.port_select.options = options

                # Try to maintain current selection, otherwise use first option
                if current_selection in options:
                    self.port_select.value = current_selection
                elif self.modbus_port in values:
                    # Find the display name for current modbus_port
                    for port in self.available_ports:
                        if port['device'] == self.modbus_port:
                            self.port_select.value = port['display_name']
                            break
                else:
                    self.port_select.value = options[0] if options else 'Manual Entry'

            ui.notify(f"Found {len(self.available_ports) - 1} serial ports", type="info")  # -1 for manual entry

        except Exception as e:
            logger.error(f"Error refreshing serial ports: {e}")
            ui.notify(f"Error refreshing ports: {e}", type="negative")

    def validate_serial_port(self, port: str) -> bool:
        """Validate if a serial port is accessible"""
        if port == 'manual':
            return True

        try:
            import serial
            import os

            # Check if the port device exists (for Linux)
            if port.startswith('/dev/'):
                if not os.path.exists(port):
                    logger.warning(f"Port device {port} does not exist")
                    return False

            # Try to open the port briefly to check if it's available
            with serial.Serial(port, timeout=0.1) as test_port:
                pass
            return True
        except serial.SerialException as e:
            logger.warning(f"Port {port} validation failed (SerialException): {e}")
            return False
        except PermissionError as e:
            logger.warning(f"Port {port} validation failed (Permission denied): {e}")
            return False
        except Exception as e:
            logger.warning(f"Port {port} validation failed: {e}")
            return False

    def on_port_selection_change(self, e):
        """Handle port selection change"""
        try:
            selected_display = e.value

            # Find the corresponding port device
            selected_port = None
            for port in self.available_ports:
                if port['display_name'] == selected_display:
                    selected_port = port
                    break

            if selected_port:
                if selected_port['device'] == 'manual':
                    # Show manual input
                    self.manual_port_input.visible = True
                    self.update_port_info_display(selected_port)
                else:
                    # Hide manual input and update modbus_port
                    self.manual_port_input.visible = False
                    self.modbus_port = selected_port['device']
                    self.update_port_info_display(selected_port)

        except Exception as e:
            logger.error(f"Error handling port selection: {e}")

    def update_port_info_display(self, port_info: Dict[str, str]):
        """Update the port information display"""
        try:
            if port_info['device'] == 'manual':
                info_html = '''<div class="text-caption text-grey-6">
                    Enter port manually:<br>
                    • Windows: COM1, COM2, COM3, etc.<br>
                    • Linux: /dev/ttyACM1, /dev/ttyCH340USB1, /dev/ttyUSB0<br>
                    • Common RS485: /dev/ttyRS485*
                </div>'''
            else:
                info_html = f'''
                <div class="text-caption text-grey-6">
                    <div><strong>Device:</strong> {port_info['device']}</div>
                    <div><strong>Description:</strong> {port_info['description']}</div>
                    <div><strong>Hardware ID:</strong> {port_info['hwid']}</div>
                    <div><strong>Manufacturer:</strong> {port_info['manufacturer']}</div>
                </div>
                '''

            if hasattr(self, 'port_info_display'):
                self.port_info_display.content = info_html

        except Exception as e:
            logger.error(f"Error updating port info display: {e}")

    def test_selected_port(self):
        """Test the currently selected port"""
        try:
            current_port = self.modbus_port

            if hasattr(self, 'port_select') and self.port_select.value:
                # Get port from selection
                for port in self.available_ports:
                    if port['display_name'] == self.port_select.value:
                        if port['device'] != 'manual':
                            current_port = port['device']
                        break

            if not current_port:
                ui.notify("Please select or enter a port", type="warning")
                return

            # Test the port
            if self.validate_serial_port(current_port):
                ui.notify(f"Port {current_port} is accessible", type="positive")
            else:
                ui.notify(f"Port {current_port} is not accessible or in use", type="negative")

        except Exception as e:
            logger.error(f"Error testing port: {e}")
            ui.notify(f"Error testing port: {e}", type="negative")

    def get_selected_port(self) -> str:
        """Get the currently selected port"""
        if hasattr(self, 'port_select') and self.port_select.value:
            for port in self.available_ports:
                if port['display_name'] == self.port_select.value:
                    if port['device'] == 'manual':
                        return self.modbus_port  # Use manually entered port
                    else:
                        return port['device']
        return self.modbus_port

    async def update_connection_status(self):
        # Update header status (compact)
        if self.connection_status:
            daq_status = "Connected" if self.daq and self.daq.is_connected() else "Disconnected"

            vlt_status = "Disconnected"
            vlt_port_info = ""
            if self.vlt_controller and self.vlt_controller.is_connected():
                vlt_status = "Connected"
                vlt_port_info = f" ({self.vlt_controller.port})"

            adam_status = "Connected" if self.adam4024_controller and self.adam4024_controller.is_connected() else "Disconnected"

            self.connection_status.clear()
            with self.connection_status:
                ui.label(f"DAQ970A: {daq_status}").classes('text-white text-caption')
                ui.label(f"VLT Drives: {vlt_status}{vlt_port_info}").classes('text-white text-caption')
                ui.label(f"ADAM-4024: {adam_status}").classes('text-white text-caption')

        # Update detailed status in settings tab
        if hasattr(self, 'detailed_connection_status') and self.detailed_connection_status:
            daq_connected = bool(self.daq and self.daq.is_connected())
            vlt_connected = bool(self.vlt_controller and self.vlt_controller.is_connected())
            adam_connected = bool(self.adam4024_controller and self.adam4024_controller.is_connected())

            self.detailed_connection_status.clear()
            with self.detailed_connection_status:
                # DAQ status
                with ui.row().classes('items-center gap-2'):
                    if daq_connected:
                        ui.icon('check_circle', color='green')
                        ui.label(f"DAQ970A: Connected ({self.daq_ip})")
                    else:
                        ui.icon('cancel', color='red')
                        ui.label(f"DAQ970A: Disconnected")

                # VLT status
                with ui.row().classes('items-center gap-2'):
                    if vlt_connected:
                        ui.icon('check_circle', color='green')
                        ui.label(f"VLT Drives: Connected ({self.vlt_controller.port})")
                    else:
                        ui.icon('cancel', color='red')
                        ui.label(f"VLT Drives: Disconnected")

                # ADAM-4024 status
                with ui.row().classes('items-center gap-2'):
                    if adam_connected:
                        ui.icon('check_circle', color='green')
                        ui.label(f"ADAM-4024: Connected (Unit ID: {self.adam4024_unit_id})")
                    else:
                        ui.icon('cancel', color='red')
                        ui.label(f"ADAM-4024: Disconnected")

                # Overall status
                connected_count = sum([daq_connected, vlt_connected, adam_connected])
                if connected_count == 3:
                    ui.label("All equipment connected").classes('text-positive mt-2')
                elif connected_count > 0:
                    ui.label("Partial connection").classes('text-orange mt-2')
                else:
                    ui.label("No equipment connected").classes('text-negative mt-2')

    async def data_collection_loop(self):
        """Main data collection loop with proper error handling and UI updates"""
        while self.is_running:
            loop_start = time.time()
            try:
                timestamp = datetime.now()
                data_point = {"timestamp": timestamp}

                # Read DAQ data with timeout
                if self.daq and self.daq.is_connected():
                    try:
                        daq_data = await asyncio.wait_for(
                            self.daq.read_all_channels(),
                            timeout=5.0
                        )
                        data_point.update(daq_data)

                        # Queue UI updates to prevent blocking
                        if self.ui_update_queue:
                            await self.ui_update_queue.put(("temperature", daq_data))
                            await self.ui_update_queue.put(("flow", daq_data))

                    except asyncio.TimeoutError:
                        logger.warning("DAQ read timeout - continuing with next cycle")
                    except Exception as e:
                        logger.error(f"DAQ read error: {e}")

                # Read VLT drive data with timeout
                if self.vlt_controller and self.vlt_controller.is_connected():
                    try:
                        drive_data = await asyncio.wait_for(
                            self.vlt_controller.read_all_drives(),
                            timeout=3.0
                        )
                        for drive_id, drive in drive_data.items():
                            data_point[f"drive_{drive_id}_speed"] = drive.actual_speed
                            data_point[f"drive_{drive_id}_setpoint"] = drive.speed_setpoint
                            data_point[f"drive_{drive_id}_status"] = drive.status

                        # Queue drive UI updates
                        if self.ui_update_queue:
                            await self.ui_update_queue.put(("drives", drive_data))

                    except asyncio.TimeoutError:
                        logger.warning("VLT read timeout - continuing with next cycle")
                    except Exception as e:
                        logger.error(f"VLT read error: {e}")

                # Read ADAM-4024 valve data
                if self.adam4024_controller and self.adam4024_controller.is_connected():
                    try:
                        valve_channels = await asyncio.wait_for(
                            self.adam4024_controller.get_all_channels(),
                            timeout=2.0
                        )
                        for channel_id, channel in valve_channels.items():
                            data_point[f"valve_{channel_id}_voltage"] = channel.output_voltage
                            data_point[f"valve_{channel_id}_percent"] = channel.output_percent
                            data_point[f"valve_{channel_id}_name"] = channel.name

                        # Queue valve UI updates
                        if self.ui_update_queue:
                            await self.ui_update_queue.put(("valves", valve_channels))

                    except asyncio.TimeoutError:
                        logger.warning("ADAM-4024 read timeout - continuing with next cycle")
                    except Exception as e:
                        logger.error(f"ADAM-4024 read error: {e}")

                # Update auto control system if any loops are enabled
                loops = self.auto_control.get_all_loops()
                any_enabled = any(loop.enabled for loop in loops.values()) if loops else False

                if any_enabled:
                    # Start control system if not running and loops are enabled
                    if not self.auto_control.is_running():
                        await self.auto_control.start_control_system()

                    try:
                        control_results = await self.auto_control.update_control_loops(
                            data_point, self.vlt_controller, timestamp
                        )

                        # Add control system data to data point
                        for loop_id, status in control_results.items():
                            data_point[f"control_{loop_id}_setpoint"] = status.get('setpoint')
                            data_point[f"control_{loop_id}_output"] = status.get('output')
                            data_point[f"control_{loop_id}_error"] = status.get('error')
                            data_point[f"control_{loop_id}_enabled"] = status.get('enabled')

                            # Add control type and current sensor value
                            loop = self.auto_control.get_control_loop(loop_id)
                            if loop:
                                control_type = getattr(loop.config, 'control_type', 'temperature')
                                data_point[f"control_{loop_id}_type"] = control_type

                                # Store current process value (temperature or flow)
                                current_value = status.get('temperature')  # This field contains current process value
                                if control_type == 'temperature':
                                    data_point[f"control_{loop_id}_temperature"] = current_value
                                else:
                                    data_point[f"control_{loop_id}_flow"] = current_value

                        # Queue control UI updates
                        if self.ui_update_queue:
                            await self.ui_update_queue.put(("control", control_results))

                    except Exception as e:
                        logger.error(f"Auto control update error: {e}")

                elif self.auto_control.is_running() and not any_enabled:
                    # Stop control system if no loops are enabled
                    await self.auto_control.stop_control_system()

                # Store data
                self.data_history.append(data_point)

                # Keep only last 1000 points
                if len(self.data_history) > 1000:
                    self.data_history = self.data_history[-1000:]

                # Queue chart updates (less frequent to improve performance)
                if len(self.data_history) % 5 == 0 and self.ui_update_queue:  # Update charts every 5 data points
                    await self.ui_update_queue.put(("charts", None))

                # Update last activity time
                self.last_update_time = time.time()

                # Ensure we don't run too fast
                loop_duration = time.time() - loop_start
                sleep_time = max(0.1, self.update_interval - loop_duration)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(1.0)  # Wait before retrying on error

    async def ui_update_processor(self):
        """Process UI updates from queue to prevent blocking"""
        while True:
            try:
                # Check if queue is initialized
                if not self.ui_update_queue:
                    await asyncio.sleep(1.0)
                    continue

                # Wait for UI update with timeout
                update_type, data = await asyncio.wait_for(
                    self.ui_update_queue.get(),
                    timeout=1.0
                )

                if update_type == "temperature":
                    await self.update_temperature_displays(data)
                elif update_type == "flow":
                    await self.update_flow_displays(data)
                elif update_type == "drives":
                    await self.update_drive_displays(data)
                elif update_type == "valves":
                    await self.update_valve_displays(data)
                elif update_type == "control":
                    await self.update_control_displays(data)
                elif update_type == "charts":
                    await self.update_charts()

            except asyncio.TimeoutError:
                # No updates to process, continue
                continue
            except Exception as e:
                logger.error(f"Error processing UI update: {e}")
                await asyncio.sleep(0.1)

    async def keepalive_task_func(self):
        """Keepalive task to maintain UI responsiveness"""
        last_heartbeat = time.time()

        while True:
            try:
                current_time = time.time()

                # Update UI status indicator
                try:
                    if hasattr(self, 'ui_status_indicator'):
                        if current_time - last_heartbeat < 10:
                            self.ui_status_indicator.props('color=green')
                            self.ui_status_indicator.tooltip = 'UI Responsive'
                        else:
                            self.ui_status_indicator.props('color=orange')
                            self.ui_status_indicator.tooltip = 'UI Slow Response'

                        if current_time - last_heartbeat > 30:
                            self.ui_status_indicator.props('color=red')
                            self.ui_status_indicator.tooltip = 'UI Unresponsive'

                    last_heartbeat = current_time

                except Exception as ui_error:
                    logger.warning(f"UI update error in keepalive: {ui_error}")

                # Update connection status periodically
                if not hasattr(self, 'last_update_time') or self.last_update_time is None:
                    self.last_update_time = current_time
                elif current_time - self.last_update_time > 30:
                    try:
                        await self.update_connection_status()
                        self.last_update_time = current_time
                    except Exception as conn_error:
                        logger.warning(f"Connection status update error: {conn_error}")

                # Periodic memory cleanup
                if current_time % 300 < 5:  # Every 5 minutes
                    import gc
                    gc.collect()

                # Check for stale UI updates
                if self.ui_update_queue and self.ui_update_queue.qsize() > 50:
                    logger.warning(f"UI update queue is large: {self.ui_update_queue.qsize()}")
                    # Clear old updates to prevent memory buildup
                    while self.ui_update_queue.qsize() > 10:
                        try:
                            self.ui_update_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                await asyncio.sleep(5.0)  # Keepalive every 5 seconds

            except Exception as e:
                logger.error(f"Keepalive error: {e}")
                await asyncio.sleep(5.0)

    async def update_temperature_displays(self, data: Dict[str, float]):
        for tag in self.temp_config['mappings'].keys():
            if tag in data and tag in self.temp_cards:
                temp_value = data[tag]
                if not pd.isna(temp_value):
                    units = self.temp_config['mappings'][tag].get('units', '°C')
                    self.temp_cards[tag]['value'].text = f"{temp_value:.1f}{units}"
                    # Color coding based on temperature ranges
                    if temp_value > 80:
                        self.temp_cards[tag]['card'].style('background-color: #ffebee; border-left: 4px solid #f44336')
                    elif temp_value > 60:
                        self.temp_cards[tag]['card'].style('background-color: #fff3e0; border-left: 4px solid #ff9800')
                    else:
                        self.temp_cards[tag]['card'].style('background-color: #e8f5e8; border-left: 4px solid #4caf50')
                else:
                    self.temp_cards[tag]['value'].text = "N/A"

    async def update_flow_displays(self, data: Dict[str, float]):
        for tag in self.flow_config['mappings'].keys():
            if tag in data and tag in self.flow_cards:
                flow_value = data[tag]
                if not pd.isna(flow_value):
                    units = self.flow_config['mappings'][tag].get('units', '%')
                    self.flow_cards[tag]['value'].text = f"{flow_value:.1f}{units}"
                    # Color coding based on flow ranges
                    if flow_value > 80:
                        self.flow_cards[tag]['card'].style('background-color: #e8f5e8; border-left: 4px solid #4caf50')
                    elif flow_value > 20:
                        self.flow_cards[tag]['card'].style('background-color: #fff3e0; border-left: 4px solid #ff9800')
                    else:
                        self.flow_cards[tag]['card'].style('background-color: #ffebee; border-left: 4px solid #f44336')
                else:
                    self.flow_cards[tag]['value'].text = "N/A"

    async def update_drive_displays(self, drives: Dict[int, Any]):
        for drive_id, drive in drives.items():
            key = f"drive_{drive_id}"
            if key in self.drive_cards:
                card = self.drive_cards[key]
                card['speed_label'].text = f"Speed: {drive.actual_speed:.1f}%"
                card['setpoint_label'].text = f"Setpoint: {drive.speed_setpoint:.1f}%"
                card['status_label'].text = f"Status: {drive.status}"

                # Update status color
                if drive.fault:
                    card['status_label'].style('color: red')
                elif drive.status == "Running":
                    card['status_label'].style('color: green')
                else:
                    card['status_label'].style('color: orange')

    async def update_valve_displays(self, valves: Dict[int, Any]):
        """Update valve display cards with current status"""
        for valve_id, valve in valves.items():
            key = f"valve_{valve_id}"
            if hasattr(self, 'valve_cards') and key in self.valve_cards:
                card = self.valve_cards[key]
                card['voltage_label'].text = f"{valve.output_voltage:.2f}V"
                card['percent_label'].text = f"{valve.output_percent:.1f}%"
                card['name_label'].text = valve.name

                # Update status color based on output level
                if valve.output_percent > 80:
                    card['percent_label'].style('color: red; font-weight: bold')
                elif valve.output_percent > 50:
                    card['percent_label'].style('color: orange; font-weight: bold')
                elif valve.output_percent > 10:
                    card['percent_label'].style('color: green; font-weight: bold')
                else:
                    card['percent_label'].style('color: gray; font-weight: normal')

    async def update_control_displays(self, control_results: Dict[str, Any]):
        """Update auto control system displays"""
        try:
            # Update the status display
            await self.update_control_status_display()

            # Update individual loop displays is handled by update_control_status_display()

        except Exception as e:
            logger.error(f"Error updating control displays: {e}")

    async def update_charts(self):
        if len(self.data_history) < 2:
            return

        # Create dataframe from history
        df = pd.DataFrame(self.data_history)

        # Update temperature chart
        if self.temp_chart and 'timestamp' in df.columns:
            fig_temp = go.Figure()

            # Add traces for first 8 temperature channels (to avoid clutter)
            temp_tags = list(self.temp_config['mappings'].keys())[:8]
            for tag in temp_tags:
                if tag in df.columns:
                    label = self.temp_config['mappings'][tag]['label']
                    fig_temp.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[tag],
                        mode='lines',
                        name=f"{tag}: {label}",
                        line=dict(width=1)
                    ))

            fig_temp.update_layout(
                title="Temperature Trends (First 8 Channels)",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                height=300,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )

            self.temp_chart.figure = fig_temp

        # Update flow chart
        if self.flow_chart and 'timestamp' in df.columns:
            fig_flow = go.Figure()

            for tag in self.flow_config['mappings'].keys():
                if tag in df.columns:
                    label = self.flow_config['mappings'][tag]['label']
                    units = self.flow_config['mappings'][tag].get('units', '%')
                    fig_flow.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[tag],
                        mode='lines',
                        name=f"{tag}: {label}",
                        line=dict(width=2)
                    ))

            fig_flow.update_layout(
                title="Flow Meter Trends",
                xaxis_title="Time",
                yaxis_title="Flow (%)",
                height=300,
                showlegend=True
            )

            self.flow_chart.figure = fig_flow

    async def set_drive_speed(self, drive_id: int, speed: float):
        if self.vlt_controller and self.vlt_controller.is_connected():
            try:
                success = await self.vlt_controller.set_drive_speed(drive_id, speed)
                if success:
                    ui.notify(f"Drive {drive_id} speed set to {speed:.1f}%", type="positive")
                else:
                    ui.notify(f"Failed to set Drive {drive_id} speed", type="negative")
            except Exception as e:
                ui.notify(f"Error setting drive speed: {e}", type="negative")
        else:
            ui.notify("VLT drives not connected", type="warning")

    def create_ui(self):
        ui.page_title("Data Collection HMI")

        # Add client-side heartbeat script to maintain WebSocket connection
        ui.add_head_html('''
        <script>
        // Client-side keepalive for WebSocket connection
        let heartbeatInterval;

        function startHeartbeat() {
            heartbeatInterval = setInterval(() => {
                // Send a simple ping to keep connection alive
                if (window.socket && window.socket.readyState === WebSocket.OPEN) {
                    window.socket.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000); // Every 30 seconds
        }

        function stopHeartbeat() {
            if (heartbeatInterval) {
                clearInterval(heartbeatInterval);
            }
        }

        // Start heartbeat when page loads
        document.addEventListener('DOMContentLoaded', startHeartbeat);

        // Handle visibility change to prevent issues when tab is inactive
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                stopHeartbeat();
            } else {
                startHeartbeat();
            }
        });

        // Reconnection logic for WebSocket
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;

        function attemptReconnect() {
            if (reconnectAttempts < maxReconnectAttempts) {
                reconnectAttempts++;
                setTimeout(() => {
                    location.reload();
                }, 1000 * reconnectAttempts);
            }
        }

        // Fullscreen functionality
        let isFullscreen = false;

        function toggleFullscreen() {
            if (!isFullscreen) {
                // Enter fullscreen
                if (document.documentElement.requestFullscreen) {
                    document.documentElement.requestFullscreen();
                } else if (document.documentElement.webkitRequestFullscreen) {
                    document.documentElement.webkitRequestFullscreen();
                } else if (document.documentElement.msRequestFullscreen) {
                    document.documentElement.msRequestFullscreen();
                }
            } else {
                // Exit fullscreen
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                } else if (document.msExitFullscreen) {
                    document.msExitFullscreen();
                }
            }
        }

        // Listen for fullscreen changes to update button icon
        function handleFullscreenChange() {
            isFullscreen = !!(document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement);

            // Update button icon
            const fullscreenBtn = document.querySelector('.fullscreen-toggle-btn');
            if (fullscreenBtn) {
                const icon = fullscreenBtn.querySelector('i');
                if (icon) {
                    icon.textContent = isFullscreen ? 'fullscreen_exit' : 'fullscreen';
                }
            }
        }

        // Listen for fullscreen changes
        document.addEventListener('fullscreenchange', handleFullscreenChange);
        document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
        document.addEventListener('msfullscreenchange', handleFullscreenChange);

        // Make toggleFullscreen globally available
        window.toggleFullscreen = toggleFullscreen;

        // Monitor WebSocket connection
        window.addEventListener('beforeunload', stopHeartbeat);
        </script>
        ''')

        # Start keepalive task when page is ready
        ui.timer(1.0, self.start_background_tasks, once=True)

        # Header as scrollable content instead of fixed
        with ui.card().classes('w-full mb-4 p-4 bg-primary text-white'):
            with ui.row().classes('items-center justify-between w-full'):
                with ui.row().classes('items-center gap-3'):
                    # Fullscreen toggle button
                    self.fullscreen_btn = ui.button(
                        icon='fullscreen',
                        on_click=self.toggle_fullscreen
                    ).props('color=white size=md flat round').classes('fullscreen-toggle-btn').tooltip('Toggle Fullscreen')

                    with ui.column():
                        ui.label('Data Collection HMI').classes('text-h4 text-white')
                        # Test name display (shown when recording)
                        self.test_name_display = ui.label('').classes('text-caption text-white').style('display: none')

                with ui.row().classes('items-center gap-3'):
                    # Recording controls
                    self.start_recording_btn = ui.button(
                        'Start Recording',
                        on_click=self.start_recording,
                        icon='fiber_manual_record'
                    ).props('color=red size=sm')

                    self.stop_recording_btn = ui.button(
                        'Stop Recording',
                        on_click=self.stop_recording,
                        icon='stop'
                    ).props('color=red size=sm').style('display: none')

                    self.connection_status = ui.column()

                    # Add status indicator for UI responsiveness
                    self.ui_status_indicator = ui.icon('circle', color='green').tooltip('UI Responsive')

        with ui.tabs().classes('w-full') as tabs:
            temperature_tab = ui.tab('Temperature')
            flow_tab = ui.tab('Flow Meters')
            drives_tab = ui.tab('VLT Drives')
            valves_tab = ui.tab('Valve Control')
            auto_control_tab = ui.tab('Auto Control')
            trends_tab = ui.tab('Trends')
            settings_tab = ui.tab('Settings')

        with ui.tab_panels(tabs, value=temperature_tab).classes('w-full'):
            # Temperature panel
            with ui.tab_panel(temperature_tab):
                ui.label('Temperature Channels').classes('text-h5 mb-4')
                self.temp_grid = ui.grid(columns=8).classes('w-full gap-2')
                with self.temp_grid:
                    for tag, config in self.temp_config['mappings'].items():
                        with ui.card().classes('p-2') as card:
                            ui.label(tag).classes('text-subtitle2 font-bold')
                            ui.label(config['label']).classes('text-caption')
                            value_label = ui.label("N/A").classes('text-h6')

                            # Display coordinates
                            coords = config.get('coordinates', {'r': 0, 'z': 0, 'theta': 0})
                            coord_label = ui.label(
                                f"R:{coords['r']}mm Z:{coords['z']}mm θ:{coords['theta']}°"
                            ).classes('text-caption text-grey-6')

                            self.temp_cards[tag] = {
                                'card': card,
                                'value': value_label,
                                'coordinates': coord_label
                            }

            # Flow meters panel
            with ui.tab_panel(flow_tab):
                ui.label('Flow Meters').classes('text-h5 mb-4')
                self.flow_grid = ui.grid(columns=4).classes('w-full gap-4')
                with self.flow_grid:
                    for tag, config in self.flow_config['mappings'].items():
                        with ui.card().classes('p-4') as card:
                            ui.label(tag).classes('text-h6 font-bold')
                            ui.label(config['label']).classes('text-body2')
                            value_label = ui.label("N/A").classes('text-h5')
                            self.flow_cards[tag] = {'card': card, 'value': value_label}

            # VLT drives panel
            with ui.tab_panel(drives_tab):
                ui.label('VLT Drives (6)').classes('text-h5 mb-4')
                with ui.grid(columns=3).classes('w-full gap-4'):
                    for i in range(1, 7):
                        key = f"drive_{i}"
                        with ui.card().classes('p-4'):
                            ui.label(f"VLT Drive {i}").classes('text-h6')

                            card_elements = {}
                            card_elements['speed_label'] = ui.label("Speed: N/A")
                            card_elements['setpoint_label'] = ui.label("Setpoint: N/A")
                            card_elements['status_label'] = ui.label("Status: N/A")

                            with ui.row().classes('items-center mt-2'):
                                speed_input = ui.number(
                                    label='Speed (%)',
                                    value=0,
                                    min=0,
                                    max=100,
                                    step=0.1
                                ).classes('w-32')

                                ui.button(
                                    'Set',
                                    on_click=lambda drive_id=i, inp=speed_input:
                                        self.set_drive_speed(drive_id, inp.value)
                                ).props('size=sm color=primary')

                                ui.button(
                                    'Stop',
                                    on_click=lambda drive_id=i:
                                        self.set_drive_speed(drive_id, 0)
                                ).props('size=sm color=negative')

                            self.drive_cards[key] = card_elements

            # Valve control panel
            with ui.tab_panel(valves_tab):
                ui.label('Valve Control (ADAM-4024)').classes('text-h5 mb-4')

                with ui.row().classes('w-full gap-4 mb-4'):
                    ui.button(
                        'Emergency Stop All Valves',
                        on_click=self.emergency_stop_valves,
                        icon='emergency'
                    ).props('color=red size=lg')

                    ui.button(
                        'Close All Valves',
                        on_click=self.close_all_valves,
                        icon='valve'
                    ).props('color=orange size=lg')

                with ui.grid(columns=2).classes('w-full gap-4'):
                    for i in range(1, 5):
                        valve_key = f"valve_{i}"
                        with ui.card().classes('p-4') as card:
                            ui.label(f"Valve {i}").classes('text-h6 font-bold')

                            card_elements = {}
                            card_elements['card'] = card
                            card_elements['name_label'] = ui.label("Main Control Valve").classes('text-body2')
                            card_elements['type_label'] = ui.label("Flow Control").classes('text-caption text-grey-6')

                            with ui.row().classes('items-center gap-2 mt-3'):
                                card_elements['voltage_label'] = ui.label("Voltage: 0.00V").classes('text-body1')
                                card_elements['percent_label'] = ui.label("(0.0%)").classes('text-caption text-grey-6')

                            # Voltage control
                            with ui.row().classes('items-center gap-2 mt-3'):
                                card_elements['voltage_input'] = ui.number(
                                    label='Voltage (V)',
                                    value=0.0,
                                    min=0.0,
                                    max=10.0,
                                    step=0.1
                                ).classes('w-32')

                                ui.button(
                                    'Set V',
                                    on_click=lambda valve_id=i, inp=card_elements['voltage_input']:
                                        self.set_valve_voltage(valve_id, inp.value)
                                ).props('size=sm color=primary')

                            # Percentage control
                            with ui.row().classes('items-center gap-2 mt-2'):
                                card_elements['percent_input'] = ui.number(
                                    label='Percent (%)',
                                    value=0.0,
                                    min=0.0,
                                    max=100.0,
                                    step=1.0
                                ).classes('w-32')

                                ui.button(
                                    'Set %',
                                    on_click=lambda valve_id=i, inp=card_elements['percent_input']:
                                        self.set_valve_percent(valve_id, inp.value)
                                ).props('size=sm color=secondary')

                            # Quick action buttons
                            with ui.row().classes('gap-2 mt-3'):
                                ui.button(
                                    'Close',
                                    on_click=lambda valve_id=i: self.set_valve_voltage(valve_id, 0.0)
                                ).props('size=sm color=negative')

                                ui.button(
                                    'Half',
                                    on_click=lambda valve_id=i: self.set_valve_voltage(valve_id, 5.0)
                                ).props('size=sm color=warning')

                                ui.button(
                                    'Open',
                                    on_click=lambda valve_id=i: self.set_valve_voltage(valve_id, 10.0)
                                ).props('size=sm color=positive')

                            self.valve_cards[valve_key] = card_elements

            # Auto Control panel
            with ui.tab_panel(auto_control_tab):
                ui.label('Automatic Temperature Control').classes('text-h5 mb-4')

                # System control buttons
                with ui.row().classes('w-full gap-4 mb-4'):
                    ui.button(
                        'Emergency Stop All Control',
                        on_click=self.emergency_stop_control,
                        icon='emergency'
                    ).props('color=red size=lg')

                    ui.button(
                        'Start All Loops',
                        on_click=self.start_all_control_loops,
                        icon='play_arrow'
                    ).props('color=green size=sm')

                    ui.button(
                        'Stop All Loops',
                        on_click=self.stop_all_control_loops,
                        icon='stop'
                    ).props('color=orange size=sm')

                    ui.label('Configure loops in Settings tab').classes('text-caption text-grey-6 self-center ml-4')

                # System status display
                with ui.card().classes('p-4 mb-4 bg-grey-1'):
                    ui.label('Control System Status').classes('text-h6 mb-3')
                    self.control_status_display = ui.column()

                # Control loops display
                ui.label('Control Loops').classes('text-h6 mb-3')
                with ui.scroll_area().classes('h-96'):
                    self.control_loops_display = ui.column().classes('w-full gap-4')

            # Trends panel
            with ui.tab_panel(trends_tab):
                ui.label('Data Trends').classes('text-h5 mb-4')

                with ui.row().classes('w-full gap-2'):
                    ui.button(
                        'Start Logging' if not self.is_running else 'Stop Logging',
                        on_click=self.toggle_logging
                    ).props('color=primary')

                    ui.button(
                        'Export Data',
                        on_click=self.export_data
                    ).props('color=secondary')

                    ui.button(
                        'View Probe Positions',
                        on_click=self.show_coordinate_visualization
                    ).props('color=info')

                self.temp_chart = ui.plotly(go.Figure()).classes('w-full')
                self.flow_chart = ui.plotly(go.Figure()).classes('w-full')

            # Settings panel
            with ui.tab_panel(settings_tab):
                ui.label('Equipment Settings & Control').classes('text-h4 mb-4')

                # Connection control section at the top
                with ui.card().classes('p-4 mb-4 bg-grey-1'):
                    with ui.row().classes('items-start justify-between w-full'):
                        with ui.column().classes('flex-none'):
                            ui.label('Equipment Connection').classes('text-h6 mb-3')
                            with ui.row().classes('gap-3'):
                                ui.button(
                                    'Connect Equipment',
                                    on_click=self.connect_equipment,
                                    icon='link'
                                ).props('color=positive size=lg')

                                ui.button(
                                    'Disconnect Equipment',
                                    on_click=self.disconnect_equipment,
                                    icon='link_off'
                                ).props('color=negative size=lg')

                        # Connection status display
                        with ui.column().classes('flex-1 ml-6'):
                            ui.label('Connection Status').classes('text-subtitle1 mb-2')
                            self.detailed_connection_status = ui.column()

                # Equipment configuration section
                ui.label('Equipment Configuration').classes('text-h5 mb-3')

                with ui.card().classes('p-4 mb-4'):
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.icon('settings_ethernet', color='primary')
                        ui.label('DAQ970A Settings').classes('text-h6')

                    ui.input(
                        label='IP Address',
                        value=self.daq_ip,
                        on_change=lambda e: setattr(self, 'daq_ip', e.value),
                        placeholder='e.g., 192.168.1.100'
                    ).classes('w-64')

                with ui.card().classes('p-4 mb-4'):
                    with ui.row().classes('items-center gap-2 mb-3'):
                        ui.icon('electrical_services', color='primary')
                        ui.label('VLT Drive Settings').classes('text-h6')

                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('Modbus Port:').classes('w-24')

                        # Get available ports on UI creation
                        self.available_ports = self.get_available_serial_ports()
                        port_options = [port['display_name'] for port in self.available_ports]
                        port_values = [port['device'] for port in self.available_ports]

                        # Find current port in available ports or use first available
                        current_port_index = 0
                        if self.modbus_port in port_values:
                            current_port_index = port_values.index(self.modbus_port)

                        self.port_select = ui.select(
                            options=port_options,
                            value=port_options[current_port_index] if port_options else 'Manual Entry',
                            on_change=self.on_port_selection_change
                        ).classes('flex-1')

                        ui.button(
                            icon='refresh',
                            on_click=self.refresh_serial_ports
                        ).props('size=sm').tooltip('Refresh ports')

                        ui.button(
                            icon='check_circle',
                            on_click=self.test_selected_port
                        ).props('size=sm color=positive').tooltip('Test port')

                    # Manual port entry (shown when 'manual' is selected)
                    self.manual_port_input = ui.input(
                        label='Manual Port Entry',
                        value=self.modbus_port,
                        on_change=lambda e: setattr(self, 'modbus_port', e.value)
                    ).classes('w-full')

                    # Initially hide manual input if a port is selected
                    if current_port_index > 0:  # Not manual entry
                        self.manual_port_input.visible = False

                    # Port information display
                    self.port_info_display = ui.html('<div class="text-caption text-grey-6">Select a port to see details</div>')

                    # Update port info display
                    if current_port_index < len(self.available_ports):
                        self.update_port_info_display(self.available_ports[current_port_index])

                    # VLT Unit ID Configuration
                    ui.label('Drive Unit IDs:').classes('text-subtitle2 mt-4 mb-2')
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Configure Modbus Unit IDs for each VLT drive').classes('text-body2 text-grey-7')
                        ui.button(
                            'Configure Unit IDs',
                            on_click=self.show_vlt_config,
                            icon='settings'
                        ).props('size=sm color=secondary')

                    # Display current unit IDs
                    unit_ids_display = ', '.join(map(str, self.vlt_unit_ids))
                    self.vlt_unit_ids_label = ui.label(f'Current Unit IDs: {unit_ids_display}').classes('text-caption text-grey-6 mt-1')

                with ui.card().classes('p-4 mb-4'):
                    with ui.row().classes('items-center gap-2 mb-3'):
                        ui.icon('output', color='primary')
                        ui.label('ADAM-4024 Analog Output Settings').classes('text-h6')

                    ui.label('Configure analog output device for valve control').classes('text-body2 text-grey-7 mb-3')

                    with ui.row().classes('items-center gap-4'):
                        ui.number(
                            label='Modbus Unit ID',
                            value=self.adam4024_unit_id,
                            min=1,
                            max=247,
                            on_change=lambda e: setattr(self, 'adam4024_unit_id', int(e.value)) if e.value else None
                        ).classes('w-32')

                        ui.label('Channels: 4 (0-10V analog outputs)').classes('text-body2 text-grey-7')

                    with ui.row().classes('items-center gap-2 mt-3'):
                        ui.label('Valve Configuration:').classes('text-subtitle2')
                        ui.button(
                            'Configure Valves',
                            on_click=self.show_valve_config,
                            icon='settings'
                        ).props('size=sm color=secondary')

                # Channel configuration section
                ui.label('Channel Configuration').classes('text-h5 mb-3 mt-4')

                with ui.card().classes('p-4 mb-4'):
                    with ui.row().classes('items-center gap-2 mb-3'):
                        ui.icon('tune', color='primary')
                        ui.label('Dynamic Channel Configuration').classes('text-h6')

                    ui.label('Configure number of channels and their assignments').classes('text-body2 text-grey-7 mb-3')

                    # Channel count configuration
                    with ui.row().classes('items-center gap-6 mb-4'):
                        with ui.column():
                            ui.label('Temperature Channels').classes('text-subtitle2 mb-2')
                            self.temp_count_input = ui.number(
                                label='Channel Count',
                                value=len(self.temp_config['mappings']),
                                min=1,
                                max=40,
                                step=1,
                                on_change=self.update_temp_channel_count
                            ).classes('w-32')
                            self.temp_count_label = ui.label(f'Current: T01-T{len(self.temp_config["mappings"]):02d}').classes('text-caption text-grey-6')

                        with ui.column():
                            ui.label('Flow Channels').classes('text-subtitle2 mb-2')
                            self.flow_count_input = ui.number(
                                label='Channel Count',
                                value=len(self.flow_config['mappings']),
                                min=1,
                                max=10,
                                step=1,
                                on_change=self.update_flow_channel_count
                            ).classes('w-32')
                            self.flow_count_label = ui.label(f'Current: F01-F{len(self.flow_config["mappings"]):02d}').classes('text-caption text-grey-6')

                    ui.separator().classes('my-3')

                    # Channel configuration buttons
                    with ui.row().classes('gap-4'):
                        ui.button(
                            'Configure Temperature Channels',
                            on_click=self.show_temp_channel_config,
                            icon='thermostat'
                        ).props('color=primary')
                        ui.button(
                            'Configure Flow Channels',
                            on_click=self.show_flow_channel_config,
                            icon='water'
                        ).props('color=secondary')

                    # Warning about equipment lock
                    ui.label('Note: Channel count changes require disconnecting equipment').classes('text-caption text-orange-7 mt-2')

                # Test configuration section
                ui.label('Test Configuration').classes('text-h5 mb-3 mt-4')

                with ui.card().classes('p-4 mb-4'):
                    with ui.row().classes('items-center gap-2 mb-3'):
                        ui.icon('science', color='primary')
                        ui.label('Test Settings').classes('text-h6')

                    ui.label('Configure test parameters for data recording').classes('text-body2 text-grey-7 mb-3')

                    with ui.row().classes('items-center gap-4 mb-3'):
                        self.test_name_input = ui.input(
                            label='Test Name (Required)',
                            value=self.test_name,
                            placeholder='e.g., Performance_Test_001',
                            on_change=lambda e: setattr(self, 'test_name', e.value)
                        ).classes('w-64')

                        # Show lock status
                        self.test_lock_icon = ui.icon('lock_open', color='grey').tooltip('Configuration unlocked')

                    self.test_description_input = ui.textarea(
                        label='Test Description (Optional)',
                        value=self.test_description,
                        placeholder='Describe the purpose and conditions of this test...',
                        on_change=lambda e: setattr(self, 'test_description', e.value)
                    ).classes('w-full')

                    with ui.row().classes('items-center gap-4 mt-3'):
                        ui.number(
                            label='Recording Frequency (seconds)',
                            value=self.recording_frequency,
                            min=0.1,
                            max=60,
                            step=0.1,
                            on_change=lambda e: setattr(self, 'recording_frequency', e.value)
                        ).classes('w-48')

                        ui.button(
                            'View Test History',
                            on_click=self.show_test_history,
                            icon='history'
                        ).props('color=secondary size=sm')

                # Auto control configuration section
                ui.label('Auto Control Configuration').classes('text-h5 mb-3 mt-4')

                with ui.card().classes('p-4 mb-4'):
                    with ui.row().classes('items-center gap-2 mb-3'):
                        ui.icon('auto_mode', color='primary')
                        ui.label('Control Loop Settings').classes('text-h6')

                    ui.label('Configure automatic control loops (temperature or flow)').classes('text-body2 text-grey-7 mb-3')

                    with ui.row().classes('items-center gap-4 mb-3'):
                        ui.button(
                            'Configure Control Loops',
                            on_click=self.show_control_loop_config,
                            icon='settings'
                        ).props('color=primary')

                        ui.button(
                            'View Loop Status',
                            on_click=self.show_loop_status_dialog,
                            icon='visibility'
                        ).props('color=secondary')

                    # Control system settings
                    with ui.row().classes('items-center gap-4'):
                        ui.number(
                            label='Control Update Frequency (seconds)',
                            value=self.auto_control.update_frequency,
                            min=0.1,
                            max=10.0,
                            step=0.1,
                            on_change=lambda e: self.auto_control.set_update_frequency(e.value)
                        ).classes('w-48')

                        ui.number(
                            label='Max Drive Speed (%)',
                            value=self.auto_control.max_drive_speed,
                            min=0,
                            max=100,
                            step=1,
                            on_change=lambda e: self.auto_control.set_drive_speed_limits(
                                self.auto_control.min_drive_speed, e.value
                            )
                        ).classes('w-32')

                # Configuration Management section
                ui.label('Configuration Management').classes('text-h5 mb-3 mt-4')

                with ui.card().classes('p-4 mb-4'):
                    with ui.row().classes('items-center gap-2 mb-3'):
                        ui.icon('save', color='primary')
                        ui.label('Save & Load Configurations').classes('text-h6')

                    ui.label('Save current setup configuration or load a saved configuration').classes('text-body2 text-grey-7 mb-3')

                    # Save configuration section
                    with ui.row().classes('items-center gap-4 mb-4'):
                        self.config_name_input = ui.input(
                            label='Configuration Name',
                            placeholder='e.g., Production_Setup_v1',
                            validation={'Required': lambda value: value and value.strip()}
                        ).classes('w-64')

                        ui.button(
                            'Save Configuration',
                            on_click=self.save_configuration,
                            icon='save'
                        ).props('color=positive')

                    ui.separator().classes('my-3')

                    # Load configuration section
                    with ui.column().classes('w-full'):
                        with ui.row().classes('items-center gap-4 mb-3'):
                            ui.label('Load Configuration:').classes('text-subtitle2')
                            ui.button(
                                'Refresh List',
                                on_click=self.refresh_config_list,
                                icon='refresh'
                            ).props('size=sm color=secondary')

                        # Configuration list
                        self.config_list_container = ui.column().classes('w-full')
                        self.refresh_config_list()

                # Data collection settings section
                ui.label('Data Collection Settings').classes('text-h5 mb-3 mt-4')

                with ui.card().classes('p-4'):
                    with ui.row().classes('items-center gap-2 mb-3'):
                        ui.icon('schedule', color='primary')
                        ui.label('Collection Parameters').classes('text-h6')

                    ui.label('Configure data acquisition timing and behavior').classes('text-body2 text-grey-7 mb-3')

                    ui.number(
                        label='Update Interval (seconds)',
                        value=self.update_interval,
                        min=0.1,
                        max=10,
                        step=0.1,
                        on_change=lambda e: setattr(self, 'update_interval', e.value)
                    ).classes('w-48')

    def show_temp_channel_config(self):
        """Show temperature channel configuration dialog"""
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl'):
            ui.label('Temperature Channel Configuration').classes('text-h6 mb-4')

            with ui.scroll_area().classes('h-96'):
                with ui.column().classes('w-full'):
                    temp_inputs = {}
                    for tag, config in self.temp_config['mappings'].items():
                        with ui.card().classes('p-3 mb-2'):
                            with ui.row().classes('items-center gap-4 w-full'):
                                ui.label(tag).classes('text-subtitle1 font-bold w-16')

                                with ui.column().classes('flex-1'):
                                    with ui.row().classes('items-center gap-2 mb-2'):
                                        temp_inputs[tag] = {
                                            'channel': ui.number(
                                                label='DAQ Channel',
                                                value=config['channel'],
                                                min=101,
                                                max=132
                                            ).classes('w-32'),
                                            'label': ui.input(
                                                label='Description',
                                                value=config['label']
                                            ).classes('flex-1'),
                                            'units': ui.input(
                                                label='Units',
                                                value=config.get('units', '°C')
                                            ).classes('w-20')
                                        }

                                    # Coordinate inputs
                                    with ui.row().classes('items-center gap-2'):
                                        ui.label('Position:').classes('text-caption w-16')
                                        coords = config.get('coordinates', {'r': 0, 'z': 0, 'theta': 0})
                                        temp_inputs[tag]['r'] = ui.number(
                                            label='R (mm)',
                                            value=coords['r'],
                                            min=0,
                                            max=1000,
                                            step=1
                                        ).classes('w-24')
                                        temp_inputs[tag]['z'] = ui.number(
                                            label='Z (mm)',
                                            value=coords['z'],
                                            min=0,
                                            max=2000,
                                            step=1
                                        ).classes('w-24')
                                        temp_inputs[tag]['theta'] = ui.number(
                                            label='θ (°)',
                                            value=coords['theta'],
                                            min=0,
                                            max=359,
                                            step=1
                                        ).classes('w-24')

            with ui.row().classes('justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('color=grey')
                ui.button(
                    'Save',
                    on_click=lambda: self.save_temp_config(temp_inputs, dialog)
                ).props('color=primary')

        dialog.open()

    def show_flow_channel_config(self):
        """Show flow channel configuration dialog"""
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl'):
            ui.label('Flow Channel Configuration').classes('text-h6 mb-4')

            with ui.scroll_area().classes('h-96'):
                with ui.column().classes('w-full'):
                    flow_inputs = {}
                    for tag, config in self.flow_config['mappings'].items():
                        with ui.card().classes('p-3 mb-2'):
                            with ui.row().classes('items-center gap-4 w-full'):
                                ui.label(tag).classes('text-subtitle1 font-bold w-16')

                                flow_inputs[tag] = {
                                    'channel': ui.number(
                                        label='DAQ Channel',
                                        value=config['channel'],
                                        min=201,
                                        max=220
                                    ).classes('w-32'),
                                    'label': ui.input(
                                        label='Description',
                                        value=config['label']
                                    ).classes('flex-1'),
                                    'units': ui.input(
                                        label='Units',
                                        value=config.get('units', '%')
                                    ).classes('w-20'),
                                    'scale': ui.number(
                                        label='Scale',
                                        value=config.get('scale', 1.0),
                                        step=0.1,
                                        min=0.1,
                                        max=10.0
                                    ).classes('w-24')
                                }

            # Flow configuration settings
            with ui.card().classes('p-3 mt-4'):
                ui.label('Current Loop Settings').classes('text-subtitle2 mb-2')
                with ui.row().classes('items-center gap-4'):
                    min_current_input = ui.number(
                        label='Min Current (mA)',
                        value=self.flow_config.get('min_current', 4),
                        min=0,
                        max=20
                    ).classes('w-32')
                    max_current_input = ui.number(
                        label='Max Current (mA)',
                        value=self.flow_config.get('max_current', 20),
                        min=4,
                        max=20
                    ).classes('w-32')
                    range_input = ui.number(
                        label='Range (A)',
                        value=self.flow_config.get('range', 0.02),
                        step=0.001,
                        min=0.001,
                        max=0.1
                    ).classes('w-32')

            with ui.row().classes('justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('color=grey')
                ui.button(
                    'Save',
                    on_click=lambda: self.save_flow_config(
                        flow_inputs, min_current_input, max_current_input, range_input, dialog
                    )
                ).props('color=primary')

        dialog.open()

    def save_temp_config(self, inputs, dialog):
        """Save temperature channel configuration"""
        try:
            new_mappings = {}
            for tag, input_dict in inputs.items():
                new_mappings[tag] = {
                    'channel': int(input_dict['channel'].value),
                    'label': input_dict['label'].value,
                    'units': input_dict['units'].value,
                    'coordinates': {
                        'r': input_dict['r'].value,
                        'z': input_dict['z'].value,
                        'theta': input_dict['theta'].value
                    }
                }

            self.temp_config['mappings'] = new_mappings

            # Update DAQ if connected
            if self.daq:
                self.daq.update_config(temp_config=self.temp_config)

            ui.notify('Temperature channel configuration saved', type='positive')
            dialog.close()

        except Exception as e:
            ui.notify(f'Error saving configuration: {e}', type='negative')

    def save_flow_config(self, inputs, min_current_input, max_current_input, range_input, dialog):
        """Save flow channel configuration"""
        try:
            new_mappings = {}
            for tag, input_dict in inputs.items():
                new_mappings[tag] = {
                    'channel': int(input_dict['channel'].value),
                    'label': input_dict['label'].value,
                    'units': input_dict['units'].value,
                    'scale': float(input_dict['scale'].value)
                }

            self.flow_config['mappings'] = new_mappings
            self.flow_config['min_current'] = min_current_input.value
            self.flow_config['max_current'] = max_current_input.value
            self.flow_config['range'] = range_input.value

            # Update DAQ if connected
            if self.daq:
                self.daq.update_config(flow_config=self.flow_config)

            ui.notify('Flow channel configuration saved', type='positive')
            dialog.close()

        except Exception as e:
            ui.notify(f'Error saving configuration: {e}', type='negative')

    def show_valve_config(self):
        """Show valve configuration dialog"""
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl'):
            ui.label('Valve Configuration').classes('text-h6 mb-4')

            if not self.adam4024_controller:
                ui.label('ADAM-4024 controller not initialized').classes('text-negative')
                with ui.row().classes('justify-end mt-4'):
                    ui.button('Close', on_click=dialog.close).props('color=grey')
                dialog.open()
                return

            with ui.scroll_area().classes('h-96'):
                with ui.column().classes('w-full'):
                    valve_inputs = {}
                    # Get current channel configuration synchronously
                    channels = self.adam4024_controller.channels if self.adam4024_controller else {}

                    for channel_id in range(1, 5):  # 4 channels
                        channel = channels.get(channel_id) if channels else None
                        channel_name = channel.name if channel else f"Valve {channel_id}"
                        channel_type = channel.valve_type if channel else "Control Valve"
                        min_voltage = channel.min_voltage if channel else 0.0
                        max_voltage = channel.max_voltage if channel else 10.0

                        with ui.card().classes('p-3 mb-2'):
                            with ui.column().classes('w-full'):
                                ui.label(f'Channel {channel_id}').classes('text-subtitle1 font-bold mb-2')

                                with ui.row().classes('items-center gap-4 w-full'):
                                    valve_inputs[channel_id] = {
                                        'name': ui.input(
                                            label='Valve Name',
                                            value=channel_name
                                        ).classes('flex-1'),
                                        'type': ui.select(
                                            options=['Control Valve', 'Shut-off Valve', 'Relief Valve', 'Check Valve'],
                                            value=channel_type,
                                            label='Valve Type'
                                        ).classes('w-48'),
                                        'min_voltage': ui.number(
                                            label='Min Voltage (V)',
                                            value=min_voltage,
                                            min=0,
                                            max=10,
                                            step=0.1
                                        ).classes('w-32'),
                                        'max_voltage': ui.number(
                                            label='Max Voltage (V)',
                                            value=max_voltage,
                                            min=0,
                                            max=10,
                                            step=0.1
                                        ).classes('w-32')
                                    }

            with ui.row().classes('justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('color=grey')
                ui.button(
                    'Save',
                    on_click=lambda: self.save_valve_config(valve_inputs, dialog)
                ).props('color=primary')

        dialog.open()

    def save_valve_config(self, inputs, dialog):
        """Save valve configuration"""
        try:
            if not self.adam4024_controller:
                ui.notify('ADAM-4024 controller not available', type='negative')
                return

            for channel_id, input_dict in inputs.items():
                self.adam4024_controller.configure_channel(
                    channel=channel_id,
                    name=input_dict['name'].value,
                    valve_type=input_dict['type'].value,
                    min_voltage=input_dict['min_voltage'].value,
                    max_voltage=input_dict['max_voltage'].value
                )

            ui.notify('Valve configuration saved', type='positive')
            dialog.close()

        except Exception as e:
            ui.notify(f'Error saving valve configuration: {e}', type='negative')

    def show_vlt_config(self):
        """Show VLT drive unit ID configuration dialog"""
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-3xl'):
            ui.label('VLT Drive Unit ID Configuration').classes('text-h6 mb-4')

            ui.label('Configure Modbus Unit IDs for each VLT drive. Each drive must have a unique ID (1-247).').classes('text-body2 text-grey-7 mb-4')

            with ui.scroll_area().classes('h-64'):
                with ui.column().classes('w-full'):
                    unit_id_inputs = []

                    for i, unit_id in enumerate(self.vlt_unit_ids):
                        with ui.card().classes('p-3 mb-2'):
                            with ui.row().classes('items-center gap-4 w-full'):
                                ui.label(f'Drive {i+1}:').classes('text-subtitle1 font-bold w-16')

                                unit_id_input = ui.number(
                                    label='Unit ID',
                                    value=unit_id,
                                    min=1,
                                    max=247
                                ).classes('w-24')

                                unit_id_inputs.append(unit_id_input)

                                ui.label(f'(Currently: ID {unit_id})').classes('text-caption text-grey-6 flex-1')

            # Add/Remove drive controls
            with ui.card().classes('p-3 mt-4'):
                ui.label('Drive Count Configuration').classes('text-subtitle2 mb-2')
                with ui.row().classes('items-center gap-4'):
                    ui.label(f'Current drive count: {len(self.vlt_unit_ids)}').classes('text-body2')
                    ui.button(
                        'Add Drive',
                        on_click=lambda: self.add_drive_to_config(unit_id_inputs, dialog),
                        icon='add'
                    ).props('size=sm color=positive')
                    ui.button(
                        'Remove Last Drive',
                        on_click=lambda: self.remove_drive_from_config(unit_id_inputs, dialog),
                        icon='remove'
                    ).props('size=sm color=negative')

            with ui.row().classes('justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('color=grey')
                ui.button(
                    'Save',
                    on_click=lambda: self.save_vlt_config(unit_id_inputs, dialog)
                ).props('color=primary')

        dialog.open()

    def save_vlt_config(self, unit_id_inputs, dialog):
        """Save VLT unit ID configuration"""
        try:
            # Get new unit IDs from inputs
            new_unit_ids = []
            for unit_input in unit_id_inputs:
                unit_id = int(unit_input.value)
                if unit_id < 1 or unit_id > 247:
                    ui.notify(f'Unit ID {unit_id} is out of range (1-247)', type='negative')
                    return
                new_unit_ids.append(unit_id)

            # Check for duplicates
            if len(new_unit_ids) != len(set(new_unit_ids)):
                ui.notify('Duplicate unit IDs are not allowed', type='negative')
                return

            # Update configuration
            self.vlt_unit_ids = new_unit_ids

            # Update VLT controller if it exists
            if self.vlt_controller:
                self.vlt_controller.configure_unit_ids(self.vlt_unit_ids)

            # Update display label
            unit_ids_display = ', '.join(map(str, self.vlt_unit_ids))
            if hasattr(self, 'vlt_unit_ids_label'):
                self.vlt_unit_ids_label.text = f'Current Unit IDs: {unit_ids_display}'

            ui.notify(f'VLT drive configuration saved: {len(self.vlt_unit_ids)} drives with IDs {unit_ids_display}', type='positive')
            dialog.close()

        except Exception as e:
            ui.notify(f'Error saving VLT configuration: {e}', type='negative')

    async def start_recording(self):
        """Start data recording"""
        try:
            # Validate test name
            if not self.test_name or self.test_name.strip() == "":
                ui.notify('Test name is required to start recording', type='negative')
                return

            # Check if equipment is connected
            if not self.is_any_equipment_connected():
                ui.notify('Please connect to equipment before starting recording', type='warning')
                return

            # Start the recording
            success = self.data_recorder.start_test(
                test_name=self.test_name.strip(),
                description=self.test_description,
                recording_frequency=self.recording_frequency
            )

            if success:
                # Lock equipment parameters
                self.equipment_locked = True
                self.update_parameter_lock_status()

                # Update UI
                self.start_recording_btn.style('display: none')
                self.stop_recording_btn.style('display: inline-flex')
                self.test_name_display.text = f'Recording: {self.test_name}'
                self.test_name_display.style('display: block')

                # Start data recording task
                self.recording_task = asyncio.create_task(self.data_recording_loop())

                ui.notify(f'Started recording test: {self.test_name}', type='positive')
            else:
                ui.notify('Failed to start recording. Test name may already exist.', type='negative')

        except Exception as e:
            ui.notify(f'Error starting recording: {e}', type='negative')

    async def stop_recording(self):
        """Stop data recording"""
        try:
            # Stop the recording
            success = self.data_recorder.stop_test()

            if success:
                # Unlock equipment parameters
                self.equipment_locked = False
                self.update_parameter_lock_status()

                # Stop recording task
                if hasattr(self, 'recording_task') and self.recording_task:
                    self.recording_task.cancel()

                # Update UI
                self.start_recording_btn.style('display: inline-flex')
                self.stop_recording_btn.style('display: none')
                self.test_name_display.style('display: none')

                ui.notify('Recording stopped successfully', type='positive')
            else:
                ui.notify('Failed to stop recording', type='negative')

        except Exception as e:
            ui.notify(f'Error stopping recording: {e}', type='negative')

    async def data_recording_loop(self):
        """Dedicated loop for recording data to database"""
        while self.data_recorder.is_recording():
            try:
                # Create data point from current state
                data_point = {"timestamp": datetime.now()}

                # Collect DAQ data
                if self.daq and self.daq.is_connected():
                    try:
                        daq_data = await asyncio.wait_for(
                            self.daq.read_all_channels(),
                            timeout=3.0
                        )
                        data_point.update(daq_data)
                    except Exception as e:
                        logger.warning(f"DAQ read error in recording: {e}")

                # Collect VLT drive data
                if self.vlt_controller and self.vlt_controller.is_connected():
                    try:
                        drive_data = await asyncio.wait_for(
                            self.vlt_controller.read_all_drives(),
                            timeout=2.0
                        )
                        for drive_id, drive in drive_data.items():
                            data_point[f"drive_{drive_id}_speed"] = drive.actual_speed
                            data_point[f"drive_{drive_id}_setpoint"] = drive.speed_setpoint
                            data_point[f"drive_{drive_id}_status"] = drive.status
                    except Exception as e:
                        logger.warning(f"VLT read error in recording: {e}")

                # Collect ADAM-4024 valve data
                if self.adam4024_controller and self.adam4024_controller.is_connected():
                    try:
                        valve_channels = await asyncio.wait_for(
                            self.adam4024_controller.get_all_channels(),
                            timeout=1.0
                        )
                        for channel_id, channel in valve_channels.items():
                            data_point[f"valve_{channel_id}_voltage"] = channel.output_voltage
                            data_point[f"valve_{channel_id}_percent"] = channel.output_percent
                            data_point[f"valve_{channel_id}_name"] = channel.name
                    except Exception as e:
                        logger.warning(f"ADAM-4024 read error in recording: {e}")

                # Record the data point
                self.data_recorder.record_data_point(data_point)

                # Wait for next recording interval
                await asyncio.sleep(self.recording_frequency)

            except asyncio.CancelledError:
                logger.info("Data recording loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in data recording loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry

    def is_any_equipment_connected(self) -> bool:
        """Check if any equipment is connected"""
        return (
            (self.daq and self.daq.is_connected()) or
            (self.vlt_controller and self.vlt_controller.is_connected()) or
            (self.adam4024_controller and self.adam4024_controller.is_connected())
        )

    def update_parameter_lock_status(self):
        """Update UI to show parameter lock status"""
        if hasattr(self, 'test_lock_icon'):
            if self.equipment_locked:
                self.test_lock_icon.name = 'lock'
                self.test_lock_icon.props('color=orange')
                self.test_lock_icon.tooltip = 'Configuration locked during recording'
            else:
                self.test_lock_icon.name = 'lock_open'
                self.test_lock_icon.props('color=grey')
                self.test_lock_icon.tooltip = 'Configuration unlocked'

        # Disable/enable input fields based on lock status
        if hasattr(self, 'test_name_input'):
            self.test_name_input.props(f'disable={self.equipment_locked}')
        if hasattr(self, 'test_description_input'):
            self.test_description_input.props(f'disable={self.equipment_locked}')

    def show_test_history(self):
        """Show test history dialog"""
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-5xl'):
            ui.label('Test Recording History').classes('text-h6 mb-4')

            tests = self.data_recorder.get_test_list()

            if not tests:
                ui.label('No test recordings found').classes('text-center text-grey-6 p-8')
            else:
                with ui.scroll_area().classes('h-96'):
                    with ui.column().classes('w-full'):
                        for test in tests:
                            with ui.card().classes('p-3 mb-2'):
                                with ui.row().classes('items-center justify-between w-full'):
                                    with ui.column():
                                        ui.label(test['test_name']).classes('text-subtitle1 font-bold')
                                        ui.label(f"Started: {test['start_time']}").classes('text-caption text-grey-6')
                                        if test['end_time']:
                                            ui.label(f"Ended: {test['end_time']}").classes('text-caption text-grey-6')
                                        if test['description']:
                                            ui.label(test['description']).classes('text-body2 mt-1')

                                    with ui.column().classes('items-end'):
                                        status_color = 'green' if test['status'] == 'completed' else 'orange'
                                        ui.label(test['status'].title()).classes(f'text-{status_color}')
                                        ui.label(f"Freq: {test['recording_frequency']}s").classes('text-caption')

            with ui.row().classes('justify-end mt-4'):
                ui.button('Close', on_click=dialog.close).props('color=primary')

        dialog.open()


    async def emergency_stop_control(self):
        """Emergency stop all control loops and equipment"""
        try:
            # Stop auto control system
            await self.auto_control.emergency_stop()
            self.control_enabled = False

            # Emergency stop all VLT drives
            if self.vlt_controller and self.vlt_controller.is_connected():
                await self.vlt_controller.stop_all_drives()

            # Emergency stop all valves
            if self.adam4024_controller and self.adam4024_controller.is_connected():
                await self.adam4024_controller.emergency_stop()

            ui.notify('EMERGENCY STOP - All control loops and equipment stopped', type='warning')
            await self.update_control_status_display()

        except Exception as e:
            ui.notify(f'Error during emergency stop: {e}', type='negative')

    def show_control_loop_config(self):
        """Show control loop configuration dialog"""
        # Validate flow control is working (run once)
        if not hasattr(self, '_flow_validation_done'):
            validation_passed = self.validate_flow_control_config()
            if validation_passed:
                logger.info("Flow control system validated successfully")
            self._flow_validation_done = True

        with ui.dialog() as dialog, ui.card().classes('w-full max-w-6xl'):
            ui.label('Control Loop Configuration').classes('text-h6 mb-4')

            # Get available temperature probes, flow meters, and drives
            temp_probes = list(self.temp_config['mappings'].keys())
            flow_meters = list(self.flow_config['mappings'].keys())
            drive_ids = self.vlt_unit_ids.copy()

            # Current loops display
            current_loops = self.auto_control.get_all_loops()

            with ui.scroll_area().classes('h-96'):
                with ui.column().classes('w-full'):

                    # Add new loop section
                    with ui.card().classes('p-4 mb-4 bg-blue-1'):
                        ui.label('Add New Control Loop').classes('text-subtitle1 font-bold mb-3')

                        with ui.row().classes('items-end gap-4'):
                            loop_id_input = ui.input(
                                label='Loop ID',
                                placeholder='e.g., LOOP1'
                            ).classes('w-32')

                            loop_name_input = ui.input(
                                label='Loop Name',
                                placeholder='e.g., Main Temperature Control'
                            ).classes('w-64')

                            # Control type selection
                            control_type_select = ui.select(
                                label='Control Type',
                                options=['temperature', 'flow'],
                                value='temperature'
                            ).classes('w-32')

                            setpoint_input = ui.number(
                                label='Target Value',
                                value=25.0,
                                min=-50,
                                max=200,
                                step=0.1
                            ).classes('w-48')

                            # Update setpoint label based on control type
                            def update_setpoint_label():
                                if control_type_select.value == 'temperature':
                                    setpoint_input.label = 'Target Temperature (°C)'
                                    setpoint_input.min = -50
                                    setpoint_input.max = 200
                                    setpoint_input.step = 0.1
                                else:
                                    setpoint_input.label = 'Target Flow (%)'
                                    setpoint_input.min = 0
                                    setpoint_input.max = 100
                                    setpoint_input.step = 1.0
                                setpoint_input.update()

                            control_type_select.on_value_change = lambda: update_setpoint_label()

                        with ui.row().classes('items-start gap-4 mt-3'):
                            # Sensor selection (temperature probes or flow meters)
                            with ui.column():
                                sensor_label = ui.label('Temperature Probes').classes('text-subtitle2')
                                sensor_column = ui.column()

                                probe_checkboxes = {}
                                flow_checkboxes = {}

                                # Initialize temperature probe checkboxes
                                with sensor_column:
                                    for probe in temp_probes[:12]:  # Limit display
                                        probe_checkboxes[probe] = ui.checkbox(probe, value=False).classes('text-caption')

                                # Function to update sensor selection based on control type
                                def update_sensor_selection():
                                    sensor_column.clear()
                                    if control_type_select.value == 'temperature':
                                        sensor_label.text = 'Temperature Probes'
                                        with sensor_column:
                                            for probe in temp_probes[:12]:
                                                probe_checkboxes[probe] = ui.checkbox(probe, value=False).classes('text-caption')
                                    else:
                                        sensor_label.text = 'Flow Meters'
                                        with sensor_column:
                                            for meter in flow_meters:
                                                flow_checkboxes[meter] = ui.checkbox(meter, value=False).classes('text-caption')

                                control_type_select.on_value_change = lambda: (update_setpoint_label(), update_sensor_selection())

                            # Drive selection
                            with ui.column():
                                ui.label('VLT Drives').classes('text-subtitle2')
                                drive_checkboxes = {}
                                for drive_id in drive_ids:
                                    drive_checkboxes[drive_id] = ui.checkbox(f'Drive {drive_id}', value=False).classes('text-caption')

                            # PID parameters
                            with ui.column():
                                ui.label('PID Parameters').classes('text-subtitle2')
                                kp_input = ui.number(label='Kp (Proportional)', value=1.0, step=0.1).classes('w-32')
                                ki_input = ui.number(label='Ki (Integral)', value=0.1, step=0.01).classes('w-32')
                                kd_input = ui.number(label='Kd (Derivative)', value=0.05, step=0.01).classes('w-32')

                            # Safety limits
                            with ui.column():
                                safety_label = ui.label('Safety Limits').classes('text-subtitle2')
                                safety_column = ui.column()

                                # Initialize with temperature limits
                                with safety_column:
                                    max_temp_input = ui.number(label='Max Temp (°C)', value=100.0, step=1.0).classes('w-32')
                                    min_temp_input = ui.number(label='Min Temp (°C)', value=-10.0, step=1.0).classes('w-32')
                                    max_speed_input = ui.number(label='Max Speed (%)', value=100.0, step=1.0).classes('w-32')

                                # Flow safety limits (initially hidden)
                                max_flow_input = ui.number(label='Max Flow (%)', value=100.0, step=1.0).classes('w-32')
                                min_flow_input = ui.number(label='Min Flow (%)', value=0.0, step=1.0).classes('w-32')
                                max_flow_input.visible = False
                                min_flow_input.visible = False

                                # Function to update safety limits based on control type
                                def update_safety_limits():
                                    if control_type_select.value == 'temperature':
                                        max_temp_input.visible = True
                                        min_temp_input.visible = True
                                        max_flow_input.visible = False
                                        min_flow_input.visible = False
                                    else:
                                        max_temp_input.visible = False
                                        min_temp_input.visible = False
                                        max_flow_input.visible = True
                                        min_flow_input.visible = True

                                control_type_select.on_value_change = lambda: (update_setpoint_label(), update_sensor_selection(), update_safety_limits())

                        ui.button(
                            'Add Control Loop',
                            on_click=lambda: self.add_control_loop(
                                loop_id_input, loop_name_input, control_type_select, setpoint_input,
                                probe_checkboxes, flow_checkboxes, drive_checkboxes,
                                kp_input, ki_input, kd_input,
                                max_temp_input, min_temp_input, max_flow_input, min_flow_input, max_speed_input, dialog
                            ),
                            icon='add'
                        ).props('color=positive size=sm')

                    # Existing loops
                    if current_loops:
                        ui.label('Existing Control Loops').classes('text-subtitle1 font-bold mb-3')
                        for loop_id, loop in current_loops.items():
                            with ui.card().classes('p-3 mb-2'):
                                with ui.row().classes('items-center justify-between w-full'):
                                    with ui.column():
                                        ui.label(f'{loop_id}: {loop.config.name}').classes('text-subtitle2 font-bold')
                                        control_type = getattr(loop.config, 'control_type', 'temperature')
                                        if control_type == 'temperature':
                                            ui.label(f'Type: Temperature Control').classes('text-caption')
                                            ui.label(f'Setpoint: {loop.config.pid_params.setpoint}°C').classes('text-caption')
                                            probes = getattr(loop.config, 'temperature_probes', []) or []
                                            ui.label(f'Probes: {", ".join(probes)}').classes('text-caption')
                                        else:
                                            ui.label(f'Type: Flow Control').classes('text-caption')
                                            ui.label(f'Setpoint: {loop.config.pid_params.setpoint}%').classes('text-caption')
                                            flows = getattr(loop.config, 'flow_meters', []) or []
                                            ui.label(f'Flow Meters: {", ".join(flows)}').classes('text-caption')
                                        ui.label(f'Drives: {", ".join(map(str, loop.config.drive_ids))}').classes('text-caption')

                                    with ui.row().classes('gap-2'):
                                        if loop.enabled:
                                            ui.button('Disable', on_click=lambda l=loop: l.disable(), icon='pause').props('size=sm color=orange')
                                        else:
                                            ui.button('Enable', on_click=lambda l=loop: l.enable(), icon='play_arrow').props('size=sm color=green')

                                        ui.button(
                                            'Remove',
                                            on_click=lambda lid=loop_id: self.remove_control_loop(lid, dialog),
                                            icon='delete'
                                        ).props('size=sm color=negative')

            with ui.row().classes('justify-end gap-2 mt-4'):
                ui.button('Close', on_click=dialog.close).props('color=primary')

        dialog.open()

    def add_control_loop(self, loop_id_input, loop_name_input, control_type_select, setpoint_input,
                        probe_checkboxes, flow_checkboxes, drive_checkboxes, kp_input, ki_input, kd_input,
                        max_temp_input, min_temp_input, max_flow_input, min_flow_input, max_speed_input, dialog):
        """Add a new control loop"""
        try:
            # Validate inputs
            loop_id = loop_id_input.value.strip()
            loop_name = loop_name_input.value.strip()

            if not loop_id or not loop_name:
                ui.notify('Loop ID and name are required', type='negative')
                return

            # Get control type and selected sensors
            control_type = control_type_select.value
            selected_drives = [drive_id for drive_id, checkbox in drive_checkboxes.items() if checkbox.value]

            if control_type == 'temperature':
                selected_probes = [probe for probe, checkbox in probe_checkboxes.items() if checkbox.value]
                selected_flows = None
                if not selected_probes:
                    ui.notify('At least one temperature probe must be selected', type='negative')
                    return
            else:  # flow control
                selected_flows = [flow for flow, checkbox in flow_checkboxes.items() if checkbox.value]
                selected_probes = None
                if not selected_flows:
                    ui.notify('At least one flow meter must be selected', type='negative')
                    return

            if not selected_drives:
                ui.notify('At least one drive must be selected', type='negative')
                return

            # Create PID parameters
            pid_params = PIDParameters(
                kp=kp_input.value,
                ki=ki_input.value,
                kd=kd_input.value,
                setpoint=setpoint_input.value,
                output_max=max_speed_input.value
            )

            # Create control loop config
            if control_type == 'temperature':
                config = ControlLoopConfig(
                    loop_id=loop_id,
                    name=loop_name,
                    control_type=control_type,
                    temperature_probes=selected_probes,
                    flow_meters=None,
                    drive_ids=selected_drives,
                    pid_params=pid_params,
                    safety_temp_max=max_temp_input.value,
                    safety_temp_min=min_temp_input.value
                )
            else:  # flow control
                config = ControlLoopConfig(
                    loop_id=loop_id,
                    name=loop_name,
                    control_type=control_type,
                    temperature_probes=None,
                    flow_meters=selected_flows,
                    drive_ids=selected_drives,
                    pid_params=pid_params,
                    safety_flow_max=max_flow_input.value,
                    safety_flow_min=min_flow_input.value
                )

            # Add to control system
            success = self.auto_control.add_control_loop(config)
            if success:
                ui.notify(f'Control loop {loop_id} added successfully', type='positive')
                dialog.close()
                self.show_control_loop_config()  # Refresh dialog
            else:
                ui.notify(f'Failed to add control loop {loop_id}', type='negative')

        except ValueError as e:
            # Configuration validation error
            ui.notify(f'Configuration error: {e}', type='negative')
        except Exception as e:
            ui.notify(f'Error adding control loop: {e}', type='negative')

    def remove_control_loop(self, loop_id: str, dialog):
        """Remove a control loop"""
        try:
            success = self.auto_control.remove_control_loop(loop_id)
            if success:
                ui.notify(f'Control loop {loop_id} removed', type='positive')
                dialog.close()
                self.show_control_loop_config()  # Refresh dialog
            else:
                ui.notify(f'Failed to remove control loop {loop_id}', type='negative')

        except Exception as e:
            ui.notify(f'Error removing control loop: {e}', type='negative')

    async def update_control_status_display(self):
        """Update the control system status display"""
        if not hasattr(self, 'control_status_display'):
            return

        try:
            status = self.auto_control.get_system_status()

            self.control_status_display.clear()
            with self.control_status_display:
                with ui.row().classes('items-center gap-2'):
                    status_color = 'green' if status['enabled'] else 'grey'
                    ui.icon('auto_mode', color=status_color)
                    ui.label(f"System: {'Running' if status['enabled'] else 'Stopped'}").classes(f'text-{status_color}')

                ui.label(f"Active Loops: {status['active_loops']} / {status['total_loops']}")

                if status['emergency_stop']:
                    ui.label("EMERGENCY STOP ACTIVE").classes('text-red font-bold')

                if status['safety_triggered']:
                    ui.label("Safety limits triggered").classes('text-orange')

            # Update individual loop displays
            await self.update_control_loops_display()

        except Exception as e:
            logger.error(f"Error updating control status display: {e}")

    async def update_control_loops_display(self):
        """Update the control loops display"""
        if not hasattr(self, 'control_loops_display'):
            return

        try:
            loops = self.auto_control.get_all_loops()

            self.control_loops_display.clear()
            with self.control_loops_display:
                if not loops:
                    ui.label('No control loops configured').classes('text-center text-grey-6 p-4')
                else:
                    for loop_id, loop in loops.items():
                        with ui.card().classes('p-3'):
                            with ui.row().classes('items-center justify-between w-full'):
                                with ui.column().classes('flex-1'):
                                    ui.label(f'{loop_id}: {loop.config.name}').classes('text-subtitle1 font-bold')

                                    status_color = 'green' if loop.enabled else 'grey'
                                    ui.label(f"Status: {'Active' if loop.enabled else 'Disabled'}").classes(f'text-{status_color}')

                                    if loop.last_temperature is not None:
                                        ui.label(f'Temperature: {loop.last_temperature:.1f}°C').classes('text-body2')

                                    ui.label(f'Setpoint: {loop.config.pid_params.setpoint}°C').classes('text-body2')
                                    ui.label(f'Output: {loop.last_output:.1f}%').classes('text-body2')

                                with ui.column().classes('items-center gap-2'):
                                    # Status icon
                                    if loop.safety_triggered:
                                        ui.icon('warning', color='red').tooltip('Safety triggered')
                                    elif loop.enabled:
                                        ui.icon('play_arrow', color='green').tooltip('Active')
                                    else:
                                        ui.icon('pause', color='grey').tooltip('Disabled')

                                    # Individual control buttons
                                    with ui.row().classes('gap-1'):
                                        if loop.enabled:
                                            ui.button(
                                                icon='pause',
                                                on_click=lambda lid=loop_id: self.stop_individual_loop(lid)
                                            ).props('size=sm color=orange').tooltip(f'Stop {loop_id}')
                                        else:
                                            ui.button(
                                                icon='play_arrow',
                                                on_click=lambda lid=loop_id: self.start_individual_loop(lid)
                                            ).props('size=sm color=green').tooltip(f'Start {loop_id}')

                                        if loop.safety_triggered:
                                            ui.button(
                                                icon='refresh',
                                                on_click=lambda lid=loop_id: self.reset_loop_safety(lid)
                                            ).props('size=sm color=blue').tooltip(f'Reset Safety {loop_id}')

        except Exception as e:
            logger.error(f"Error updating control loops display: {e}")

    async def start_individual_loop(self, loop_id: str):
        """Start an individual control loop"""
        try:
            loop = self.auto_control.get_control_loop(loop_id)
            if loop:
                if not self.is_any_equipment_connected():
                    ui.notify('Please connect to equipment before starting control loops', type='warning')
                    return

                success = loop.enable()
                if success:
                    ui.notify(f'Control loop {loop_id} started', type='positive')
                    await self.update_control_status_display()
                else:
                    ui.notify(f'Failed to start control loop {loop_id}', type='negative')
            else:
                ui.notify(f'Control loop {loop_id} not found', type='negative')

        except Exception as e:
            ui.notify(f'Error starting control loop {loop_id}: {e}', type='negative')

    async def stop_individual_loop(self, loop_id: str):
        """Stop an individual control loop"""
        try:
            loop = self.auto_control.get_control_loop(loop_id)
            if loop:
                loop.disable()
                ui.notify(f'Control loop {loop_id} stopped', type='positive')
                await self.update_control_status_display()
            else:
                ui.notify(f'Control loop {loop_id} not found', type='negative')

        except Exception as e:
            ui.notify(f'Error stopping control loop {loop_id}: {e}', type='negative')

    async def reset_loop_safety(self, loop_id: str):
        """Reset safety trigger for an individual control loop"""
        try:
            loop = self.auto_control.get_control_loop(loop_id)
            if loop:
                loop.reset_safety()
                ui.notify(f'Safety reset for control loop {loop_id}', type='positive')
                await self.update_control_status_display()
            else:
                ui.notify(f'Control loop {loop_id} not found', type='negative')

        except Exception as e:
            ui.notify(f'Error resetting safety for loop {loop_id}: {e}', type='negative')

    async def start_all_control_loops(self):
        """Start all configured control loops"""
        try:
            if not self.is_any_equipment_connected():
                ui.notify('Please connect to equipment before starting control loops', type='warning')
                return

            loops = self.auto_control.get_all_loops()
            if not loops:
                ui.notify('No control loops configured', type='warning')
                return

            started_count = 0
            for loop_id, loop in loops.items():
                if loop.enable():
                    started_count += 1

            if started_count > 0:
                # Start the control system if not already running
                if not self.auto_control.is_running():
                    await self.auto_control.start_control_system()

                ui.notify(f'Started {started_count} control loop(s)', type='positive')
                await self.update_control_status_display()
            else:
                ui.notify('No control loops could be started', type='warning')

        except Exception as e:
            ui.notify(f'Error starting control loops: {e}', type='negative')

    async def stop_all_control_loops(self):
        """Stop all control loops"""
        try:
            loops = self.auto_control.get_all_loops()
            stopped_count = 0

            for loop_id, loop in loops.items():
                if loop.enabled:
                    loop.disable()
                    stopped_count += 1

            ui.notify(f'Stopped {stopped_count} control loop(s)', type='positive')
            await self.update_control_status_display()

        except Exception as e:
            ui.notify(f'Error stopping control loops: {e}', type='negative')

    def show_loop_status_dialog(self):
        """Show detailed loop status dialog"""
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-5xl'):
            ui.label('Control Loop Status').classes('text-h6 mb-4')

            loops = self.auto_control.get_all_loops()

            if not loops:
                ui.label('No control loops configured').classes('text-center text-grey-6 p-8')
            else:
                with ui.scroll_area().classes('h-96'):
                    with ui.column().classes('w-full'):
                        for loop_id, loop in loops.items():
                            with ui.card().classes('p-4 mb-3'):
                                with ui.row().classes('items-start justify-between w-full'):
                                    with ui.column().classes('flex-1'):
                                        ui.label(f'{loop_id}: {loop.config.name}').classes('text-h6 font-bold')

                                        # Status and temperatures
                                        status_color = 'green' if loop.enabled else 'grey'
                                        ui.label(f"Status: {'Active' if loop.enabled else 'Disabled'}").classes(f'text-{status_color} text-subtitle1')

                                        if loop.last_temperature is not None:
                                            ui.label(f'Current Temperature: {loop.last_temperature:.1f}°C').classes('text-body1')

                                        ui.label(f'Target Setpoint: {loop.config.pid_params.setpoint}°C').classes('text-body1')

                                        # PID output and error
                                        ui.label(f'PID Output: {loop.last_output:.1f}%').classes('text-body2')
                                        if loop.last_temperature is not None:
                                            error = loop.config.pid_params.setpoint - loop.last_temperature
                                            ui.label(f'Control Error: {error:.1f}°C').classes('text-body2')

                                    with ui.column():
                                        ui.label('Configuration').classes('text-subtitle2 font-bold')
                                        ui.label(f'Probes: {", ".join(loop.config.temperature_probes)}').classes('text-caption')
                                        ui.label(f'Drives: {", ".join(map(str, loop.config.drive_ids))}').classes('text-caption')
                                        ui.label(f'PID: Kp={loop.config.pid_params.kp:.2f}, Ki={loop.config.pid_params.ki:.3f}, Kd={loop.config.pid_params.kd:.3f}').classes('text-caption')
                                        ui.label(f'Safety: {loop.config.safety_temp_min}°C - {loop.config.safety_temp_max}°C').classes('text-caption')

                                        if loop.safety_triggered:
                                            ui.label('SAFETY TRIGGERED').classes('text-red font-bold')

            with ui.row().classes('justify-end mt-4'):
                ui.button('Close', on_click=dialog.close).props('color=primary')

        dialog.open()

    def add_drive_to_config(self, unit_id_inputs, dialog):
        """Add a new drive to configuration (placeholder - requires dialog refresh)"""
        # Find next available unit ID
        used_ids = [int(inp.value) for inp in unit_id_inputs]
        next_id = 1
        while next_id in used_ids:
            next_id += 1

        # Add to current config temporarily
        self.vlt_unit_ids.append(next_id)

        # Close and reopen dialog to refresh
        dialog.close()
        self.show_vlt_config()

    def remove_drive_from_config(self, unit_id_inputs, dialog):
        """Remove last drive from configuration (placeholder - requires dialog refresh)"""
        if len(self.vlt_unit_ids) > 1:  # Keep at least one drive
            self.vlt_unit_ids.pop()
            # Close and reopen dialog to refresh
            dialog.close()
            self.show_vlt_config()
        else:
            ui.notify('Cannot remove the last drive. At least one drive is required.', type='warning')

    async def toggle_logging(self):
        if not self.is_running:
            if not (self.daq and self.daq.is_connected()) and not (self.vlt_controller and self.vlt_controller.is_connected()):
                ui.notify("Please connect to equipment first", type="warning")
                return

            self.is_running = True
            ui.notify("Data logging started", type="positive")

            # Start background tasks
            self.data_task = asyncio.create_task(self.data_collection_loop())

            # Only start UI processor if queue is initialized
            if self.ui_update_queue:
                self.ui_processor_task = asyncio.create_task(self.ui_update_processor())

            # Ensure background tasks are started
            if not self.background_tasks_started:
                self.start_background_tasks()

        else:
            self.is_running = False
            ui.notify("Data logging stopped", type="info")

            # Cancel background tasks
            if self.data_task and not self.data_task.done():
                self.data_task.cancel()
                try:
                    await self.data_task
                except asyncio.CancelledError:
                    pass

            if hasattr(self, 'ui_processor_task') and not self.ui_processor_task.done():
                self.ui_processor_task.cancel()
                try:
                    await self.ui_processor_task
                except asyncio.CancelledError:
                    pass

    def export_data(self):
        if not self.data_history:
            ui.notify("No data to export", type="warning")
            return

        try:
            df = pd.DataFrame(self.data_history)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Export data
            data_filename = f"data_export_{timestamp}.csv"
            df.to_csv(data_filename, index=False)

            # Export channel configuration with coordinates
            config_filename = f"channel_config_{timestamp}.csv"
            config_data = []

            # Temperature channels
            for tag, config in self.temp_config['mappings'].items():
                coords = config.get('coordinates', {'r': 0, 'z': 0, 'theta': 0})
                config_data.append({
                    'Tag': tag,
                    'Type': 'Temperature',
                    'DAQ_Channel': config['channel'],
                    'Description': config['label'],
                    'Units': config['units'],
                    'R_mm': coords['r'],
                    'Z_mm': coords['z'],
                    'Theta_deg': coords['theta']
                })

            # Flow channels
            for tag, config in self.flow_config['mappings'].items():
                config_data.append({
                    'Tag': tag,
                    'Type': 'Flow',
                    'DAQ_Channel': config['channel'],
                    'Description': config['label'],
                    'Units': config['units'],
                    'R_mm': 'N/A',
                    'Z_mm': 'N/A',
                    'Theta_deg': 'N/A'
                })

            config_df = pd.DataFrame(config_data)
            config_df.to_csv(config_filename, index=False)

            ui.notify(f"Data exported to {data_filename} and config to {config_filename}", type="positive")
        except Exception as e:
            ui.notify(f"Export failed: {e}", type="negative")

    def show_coordinate_visualization(self):
        """Show 3D visualization of temperature probe positions"""
        try:
            import plotly.graph_objects as go

            # Get coordinate data
            coordinates = []
            tags = []
            temps = []

            for tag, config in self.temp_config['mappings'].items():
                coords = config.get('coordinates', {'r': 0, 'z': 0, 'theta': 0})
                if coords['r'] > 0 or coords['z'] > 0:  # Skip origin points
                    # Convert cylindrical to cartesian coordinates
                    import math
                    theta_rad = math.radians(coords['theta'])
                    x = coords['r'] * math.cos(theta_rad)
                    y = coords['r'] * math.sin(theta_rad)
                    z = coords['z']

                    coordinates.append([x, y, z])
                    tags.append(f"{tag}: {config['label']}")

                    # Get current temperature if available
                    current_temp = "N/A"
                    if self.data_history and tag in self.data_history[-1]:
                        current_temp = f"{self.data_history[-1][tag]:.1f}°C"
                    temps.append(current_temp)

            if not coordinates:
                ui.notify("No probe coordinates configured", type="warning")
                return

            coordinates = list(zip(*coordinates))

            fig = go.Figure(data=[go.Scatter3d(
                x=coordinates[0],
                y=coordinates[1],
                z=coordinates[2],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=coordinates[2],  # Color by Z coordinate
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Height (mm)")
                ),
                text=tags,
                textposition="top center",
                hovertemplate="<b>%{text}</b><br>" +
                              "X: %{x:.0f}mm<br>" +
                              "Y: %{y:.0f}mm<br>" +
                              "Z: %{z:.0f}mm<br>" +
                              "<extra></extra>"
            )])

            fig.update_layout(
                title="Temperature Probe Positions",
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Z (mm)",
                    aspectmode='cube'
                ),
                height=600
            )

            with ui.dialog() as dialog, ui.card().classes('w-full max-w-6xl'):
                ui.label('Temperature Probe 3D Positions').classes('text-h6 mb-4')
                ui.plotly(fig).classes('w-full h-96')
                with ui.row().classes('justify-end mt-4'):
                    ui.button('Close', on_click=dialog.close).props('color=primary')

            dialog.open()

        except Exception as e:
            ui.notify(f"Visualization error: {e}", type="negative")

    async def set_valve_voltage(self, valve_id: int, voltage: float):
        """Set valve voltage (0-10V)"""
        if not self.adam4024_controller or not self.adam4024_controller.is_connected():
            ui.notify("ADAM-4024 not connected", type="negative")
            return False

        try:
            success = await self.adam4024_controller.set_channel_voltage(valve_id, voltage)
            if success:
                ui.notify(f"Valve {valve_id} set to {voltage:.2f}V", type="positive")
                return True
            else:
                ui.notify(f"Failed to set valve {valve_id} voltage", type="negative")
                return False
        except Exception as e:
            ui.notify(f"Error setting valve {valve_id}: {e}", type="negative")
            return False

    async def set_valve_percent(self, valve_id: int, percent: float):
        """Set valve percentage (0-100%)"""
        if not self.adam4024_controller or not self.adam4024_controller.is_connected():
            ui.notify("ADAM-4024 not connected", type="negative")
            return False

        try:
            success = await self.adam4024_controller.set_channel_percent(valve_id, percent)
            if success:
                ui.notify(f"Valve {valve_id} set to {percent:.1f}%", type="positive")
                return True
            else:
                ui.notify(f"Failed to set valve {valve_id} percentage", type="negative")
                return False
        except Exception as e:
            ui.notify(f"Error setting valve {valve_id}: {e}", type="negative")
            return False

    async def emergency_stop_valves(self):
        """Emergency stop - close all valves immediately"""
        if not self.adam4024_controller or not self.adam4024_controller.is_connected():
            ui.notify("ADAM-4024 not connected", type="negative")
            return False

        try:
            await self.adam4024_controller.emergency_stop()
            ui.notify("Emergency stop activated - all valves closed", type="warning")
            return True
        except Exception as e:
            ui.notify(f"Emergency stop failed: {e}", type="negative")
            return False

    async def close_all_valves(self):
        """Close all valves (set to 0V)"""
        if not self.adam4024_controller or not self.adam4024_controller.is_connected():
            ui.notify("ADAM-4024 not connected", type="negative")
            return False

        try:
            success = await self.adam4024_controller.set_all_channels_off()
            if success:
                ui.notify("All valves closed", type="positive")
                return True
            else:
                ui.notify("Failed to close all valves", type="negative")
                return False
        except Exception as e:
            ui.notify(f"Error closing valves: {e}", type="negative")
            return False

    def validate_flow_control_config(self) -> bool:
        """Validate flow control configuration - for testing purposes"""
        try:
            # Test flow control configuration
            test_config = ControlLoopConfig(
                loop_id="TEST_FLOW",
                name="Test Flow Control",
                control_type="flow",
                temperature_probes=None,
                flow_meters=["F01"],
                drive_ids=[1],
                pid_params=PIDParameters(kp=1.0, ki=0.1, kd=0.05, setpoint=50.0),
                safety_flow_max=90.0,
                safety_flow_min=10.0
            )
            logger.info("Flow control configuration validation passed")
            return True
        except Exception as e:
            logger.error(f"Flow control configuration validation failed: {e}")
            return False

    def toggle_fullscreen(self):
        """Toggle fullscreen mode using JavaScript"""
        ui.run_javascript('window.toggleFullscreen()')

    def update_temp_channel_count(self, e):
        """Update the number of temperature channels"""
        try:
            new_count = int(e.value)
            if new_count < 1 or new_count > 40:
                ui.notify('Temperature channel count must be between 1 and 40', type='negative')
                return

            if self.is_connected():
                ui.notify('Please disconnect equipment before changing channel count', type='warning')
                self.temp_count_input.value = len(self.temp_config['mappings'])
                return

            # Update configuration
            from config import generate_temperature_channels
            self.temp_config = generate_temperature_channels(new_count)

            # Update label
            self.temp_count_label.text = f'Current: T01-T{new_count:02d}'

            # Update DAQ configuration if connected
            if self.daq:
                self.daq.temp_config = self.temp_config

            # Refresh UI cards
            self.refresh_temperature_cards()

            ui.notify(f'Temperature channels updated to {new_count}', type='positive')
            logger.info(f"Temperature channel count updated to {new_count}")

        except Exception as e:
            ui.notify(f'Error updating temperature channels: {e}', type='negative')
            logger.error(f"Error updating temperature channel count: {e}")

    def update_flow_channel_count(self, e):
        """Update the number of flow channels"""
        try:
            new_count = int(e.value)
            if new_count < 1 or new_count > 10:
                ui.notify('Flow channel count must be between 1 and 10', type='negative')
                return

            if self.is_connected():
                ui.notify('Please disconnect equipment before changing channel count', type='warning')
                self.flow_count_input.value = len(self.flow_config['mappings'])
                return

            # Update configuration
            from config import generate_flow_channels
            self.flow_config = generate_flow_channels(new_count)

            # Update label
            self.flow_count_label.text = f'Current: F01-F{new_count:02d}'

            # Update DAQ configuration if connected
            if self.daq:
                self.daq.flow_config = self.flow_config

            # Refresh UI cards
            self.refresh_flow_cards()

            ui.notify(f'Flow channels updated to {new_count}', type='positive')
            logger.info(f"Flow channel count updated to {new_count}")

        except Exception as e:
            ui.notify(f'Error updating flow channels: {e}', type='negative')
            logger.error(f"Error updating flow channel count: {e}")

    def is_connected(self):
        """Check if any equipment is connected"""
        return (self.daq and hasattr(self.daq, 'connected') and self.daq.connected) or \
               (self.vlt_controller and self.vlt_controller.is_connected())

    def refresh_temperature_cards(self):
        """Refresh temperature channel UI cards after configuration change"""
        try:
            # Clear existing cards
            self.temp_cards.clear()
            if hasattr(self, 'temp_grid'):
                self.temp_grid.clear()

                # Recreate cards with new configuration
                with self.temp_grid:
                    for tag, config in self.temp_config['mappings'].items():
                        with ui.card().classes('p-2') as card:
                            ui.label(tag).classes('text-subtitle2 font-bold')
                            ui.label(config['label']).classes('text-caption')
                            value_label = ui.label("N/A").classes('text-h6')

                            # Display coordinates
                            coords = config.get('coordinates', {'r': 0, 'z': 0, 'theta': 0})
                            coord_label = ui.label(
                                f"R:{coords['r']}mm Z:{coords['z']}mm θ:{coords['theta']}°"
                            ).classes('text-caption text-grey-6')

                            self.temp_cards[tag] = {
                                'card': card,
                                'value': value_label,
                                'coordinates': coord_label
                            }

                logger.info(f"Refreshed temperature cards for {len(self.temp_cards)} channels")

        except Exception as e:
            logger.error(f"Error refreshing temperature cards: {e}")

    def refresh_flow_cards(self):
        """Refresh flow channel UI cards after configuration change"""
        try:
            # Clear existing cards
            self.flow_cards.clear()
            if hasattr(self, 'flow_grid'):
                self.flow_grid.clear()

                # Recreate cards with new configuration
                with self.flow_grid:
                    for tag, config in self.flow_config['mappings'].items():
                        with ui.card().classes('p-4') as card:
                            ui.label(tag).classes('text-h6 font-bold')
                            ui.label(config['label']).classes('text-body2')
                            value_label = ui.label("N/A").classes('text-h5')
                            self.flow_cards[tag] = {'card': card, 'value': value_label}

                logger.info(f"Refreshed flow cards for {len(self.flow_cards)} channels")

        except Exception as e:
            logger.error(f"Error refreshing flow cards: {e}")

    def save_configuration(self):
        """Save current configuration to file"""
        try:
            config_name = self.config_name_input.value.strip()
            if not config_name:
                ui.notify('Please enter a configuration name', type='negative')
                return

            # Validate configuration name (no special characters)
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', config_name):
                ui.notify('Configuration name can only contain letters, numbers, underscores, and hyphens', type='negative')
                return

            success = self.config_manager.save_configuration(self, config_name)
            if success:
                ui.notify(f'Configuration "{config_name}" saved successfully', type='positive')
                self.config_name_input.value = ''  # Clear input
                self.refresh_config_list()  # Refresh the list
            else:
                ui.notify('Failed to save configuration', type='negative')

        except Exception as e:
            ui.notify(f'Error saving configuration: {e}', type='negative')
            logger.error(f"Error in save_configuration: {e}")

    def load_configuration(self, config_name: str):
        """Load and apply configuration from file"""
        try:
            if self.is_connected():
                ui.notify('Please disconnect equipment before loading configuration', type='warning')
                return

            config_data = self.config_manager.load_configuration(config_name)
            if not config_data:
                ui.notify('Failed to load configuration', type='negative')
                return

            success = self.config_manager.apply_configuration(self, config_data)
            if success:
                # Update UI elements
                self.update_ui_from_config()
                ui.notify(f'Configuration "{config_name}" loaded successfully', type='positive')
            else:
                ui.notify('Failed to apply configuration', type='negative')

        except Exception as e:
            ui.notify(f'Error loading configuration: {e}', type='negative')
            logger.error(f"Error in load_configuration: {e}")

    def delete_configuration(self, config_name: str):
        """Delete a configuration file"""
        try:
            success = self.config_manager.delete_configuration(config_name)
            if success:
                ui.notify(f'Configuration "{config_name}" deleted', type='positive')
                self.refresh_config_list()
            else:
                ui.notify('Failed to delete configuration', type='negative')

        except Exception as e:
            ui.notify(f'Error deleting configuration: {e}', type='negative')
            logger.error(f"Error in delete_configuration: {e}")

    def refresh_config_list(self):
        """Refresh the configuration list display"""
        try:
            if not hasattr(self, 'config_list_container'):
                return

            self.config_list_container.clear()
            configs = self.config_manager.list_configurations()

            if not configs:
                with self.config_list_container:
                    ui.label('No saved configurations found').classes('text-center text-grey-6 p-4')
                return

            with self.config_list_container:
                for config in configs:
                    with ui.card().classes('p-3 mb-2'):
                        with ui.row().classes('items-center justify-between w-full'):
                            with ui.column():
                                ui.label(config['display_name']).classes('text-subtitle2 font-bold')
                                ui.label(f"Created: {config['created'][:19].replace('T', ' ')}").classes('text-caption text-grey-6')

                            with ui.row().classes('gap-2'):
                                ui.button(
                                    'Load',
                                    on_click=lambda name=config['name']: self.load_configuration(name),
                                    icon='upload'
                                ).props('size=sm color=primary')

                                ui.button(
                                    'Delete',
                                    on_click=lambda name=config['name']: self.confirm_delete_config(name),
                                    icon='delete'
                                ).props('size=sm color=negative')

        except Exception as e:
            logger.error(f"Error refreshing config list: {e}")

    def confirm_delete_config(self, config_name: str):
        """Show confirmation dialog before deleting configuration"""
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Delete Configuration "{config_name}"?').classes('text-h6 mb-4')
            ui.label('This action cannot be undone.').classes('text-body2 text-grey-7 mb-4')

            with ui.row().classes('justify-end gap-2'):
                ui.button('Cancel', on_click=dialog.close).props('color=secondary')
                ui.button(
                    'Delete',
                    on_click=lambda: (self.delete_configuration(config_name), dialog.close()),
                    icon='delete'
                ).props('color=negative')

        dialog.open()

    def update_ui_from_config(self):
        """Update UI elements after loading configuration"""
        try:
            # Update channel count displays if they exist
            if hasattr(self, 'temp_count_input'):
                self.temp_count_input.value = len(self.temp_config['mappings'])
                self.temp_count_label.text = f'Current: T01-T{len(self.temp_config["mappings"]):02d}'

            if hasattr(self, 'flow_count_input'):
                self.flow_count_input.value = len(self.flow_config['mappings'])
                self.flow_count_label.text = f'Current: F01-F{len(self.flow_config["mappings"]):02d}'

            # Update VLT unit IDs display if it exists
            if hasattr(self, 'vlt_unit_ids_label'):
                unit_ids_display = ', '.join(map(str, self.vlt_unit_ids))
                self.vlt_unit_ids_label.text = f'Current Unit IDs: {unit_ids_display}'

            logger.info("UI updated from loaded configuration")

        except Exception as e:
            logger.error(f"Error updating UI from config: {e}")

def main():
    hmi = DataCollectionHMI()
    hmi.create_ui()

    # Register disconnect handler
    app.on_disconnect(hmi.cleanup_on_disconnect)

    ui.run(
        title="Data Collection HMI",
        port=8080,
        host="0.0.0.0",
        show=True,
        reload=False,
        # WebSocket and connection settings for better idle handling
        uvicorn_logging_level='warning',  # Reduce log noise
        storage_secret='data_collection_hmi_secret',  # For session persistence
        # Add favicon and other optimizations
        favicon='🏭',
        dark=None  # Let user choose theme
    )

if __name__ == "__main__":
    main()
