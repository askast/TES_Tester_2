"""
Configuration file for the HMI application
Adjust these settings according to your hardware setup
"""

# DAQ970A Configuration
DAQ_IP_ADDRESS = "192.168.1.100"  # Change to your DAQ970A IP address
DAQ_TIMEOUT = 5000  # milliseconds

# Channel count configuration
CHANNEL_COUNTS = {
    'temperature_channels': 32,  # Number of temperature channels (1-40)
    'flow_channels': 4,          # Number of flow channels (1-10)
}

# Default temperature channel labels (can be customized)
DEFAULT_TEMP_LABELS = [
    'Reactor Inlet', 'Reactor Outlet', 'Heat Exchanger 1', 'Heat Exchanger 2',
    'Condenser Inlet', 'Condenser Outlet', 'Tank 1 Top', 'Tank 1 Bottom',
    'Tank 2 Top', 'Tank 2 Bottom', 'Pipe Section A', 'Pipe Section B',
    'Pipe Section C', 'Pipe Section D', 'Ambient Zone 1', 'Ambient Zone 2',
    'Motor 1 Bearing', 'Motor 2 Bearing', 'Motor 3 Bearing', 'Motor 4 Bearing',
    'Motor 5 Bearing', 'Motor 6 Bearing', 'Control Cabinet', 'Power Cabinet'
]

# Default flow channel labels
DEFAULT_FLOW_LABELS = [
    'Main Process Flow', 'Cooling Water Flow', 'Recycle Stream Flow', 'Purge Stream Flow',
    'Feed Stream Flow', 'Product Stream Flow', 'Waste Stream Flow', 'Bypass Flow',
    'Emergency Flow', 'Spare Flow'
]

def generate_temperature_channels(count=None, sensor_type='K'):
    """Generate temperature channel configuration dynamically"""
    if count is None:
        count = CHANNEL_COUNTS['temperature_channels']

    mappings = {}
    for i in range(1, count + 1):
        tag = f'T{i:02d}'
        channel = 100 + i  # DAQ970A slot 1xx

        # Use default label if available, otherwise generic label
        if i <= len(DEFAULT_TEMP_LABELS):
            label = DEFAULT_TEMP_LABELS[i-1]
        else:
            label = f'Temperature Sensor {i}'

        # Generate default coordinates in a cylindrical pattern
        angle = (i - 1) * (360 / max(count, 8))  # Distribute around circle
        radius = 100 + ((i - 1) // 8) * 50  # Increase radius every 8 sensors
        height = 500 + ((i - 1) % 4) * 200   # Vary height in 4 levels

        mappings[tag] = {
            'channel': channel,
            'label': label,
            'units': '°C',
            'coordinates': {'r': radius, 'z': height, 'theta': angle}
        }

    return {
        'sensor_type': sensor_type,
        'mappings': mappings
    }

def generate_flow_channels(count=None):
    """Generate flow channel configuration dynamically"""
    if count is None:
        count = CHANNEL_COUNTS['flow_channels']

    mappings = {}
    for i in range(1, count + 1):
        tag = f'F{i:02d}'
        channel = 200 + i  # DAQ970A slot 2xx

        # Use default label if available, otherwise generic label
        if i <= len(DEFAULT_FLOW_LABELS):
            label = DEFAULT_FLOW_LABELS[i-1]
        else:
            label = f'Flow Meter {i}'

        mappings[tag] = {
            'channel': channel,
            'label': label,
            'units': '%',
            'scale': 1.0
        }

    return {
        'range': 0.02,         # 20mA range
        'min_current': 4,      # 4mA minimum
        'max_current': 20,     # 20mA maximum
        'mappings': mappings
    }

# Generate initial channel configurations
TEMPERATURE_CHANNELS = generate_temperature_channels()
FLOW_CHANNELS = generate_flow_channels()

# Flow channel configuration is now generated dynamically above

# VLT Drive Configuration
VLT_CONFIG = {
    'port': 'COM1',        # Serial port for Modbus communication
    'baudrate': 9600,      # Baud rate
    'parity': 'E',         # Even parity
    'stopbits': 1,
    'bytesize': 8,
    'timeout': 3,          # seconds
    'drive_count': 6,      # Number of VLT drives
    'unit_ids': [1, 2, 3, 4, 5, 6],  # Modbus unit IDs for each drive
}

# Modbus register mapping for VLT drives
VLT_REGISTERS = {
    'speed_reference': 1,     # Speed setpoint register
    'actual_speed': 102,      # Actual speed feedback register
    'status_word': 3,         # Status word register
    'control_word': 0,        # Control word register
}

# HMI Configuration
HMI_CONFIG = {
    'port': 8080,
    'host': '0.0.0.0',
    'update_interval': 1.0,   # seconds
    'data_history_limit': 1000,  # Maximum data points to keep in memory
    'auto_connect': False,    # Automatically connect on startup
}

# Alarm thresholds
ALARM_THRESHOLDS = {
    'temperature': {
        'high': 80.0,      # °C
        'warning': 60.0,   # °C
    },
    'flow': {
        'low': 20.0,       # %
        'high': 80.0,      # %
    }
}

# Data logging configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'data_export_format': 'csv',
    'export_directory': './exports',
}