import pyvisa
import asyncio
import logging
from typing import Dict, List, Optional
import time
from config import TEMPERATURE_CHANNELS, FLOW_CHANNELS

class DAQ970A:
    def __init__(self, ip_address: str, timeout: int = 5000, temp_config: Dict = None, flow_config: Dict = None):
        self.ip_address = ip_address
        self.timeout = timeout
        self.instrument = None
        self.connected = False
        self.logger = logging.getLogger(__name__)

        # Use provided config or default from config.py
        self.temp_config = temp_config or TEMPERATURE_CHANNELS
        self.flow_config = flow_config or FLOW_CHANNELS

    async def connect(self) -> bool:
        try:
            rm = pyvisa.ResourceManager('@py')
            resource_string = f'TCPIP0::{self.ip_address}::5025::SOCKET'
            self.instrument = rm.open_resource(resource_string)
            self.instrument.timeout = self.timeout
            self.instrument.read_termination = '\n'
            self.instrument.write_termination = '\n'

            # Test connection
            idn = await asyncio.to_thread(self.instrument.query, '*IDN?')
            self.logger.info(f"Connected to DAQ970A: {idn.strip()}")
            self.connected = True

            # Configure scan list based on channel mappings
            temp_channels = []
            flow_channels = []

            # Build temperature channel list from config
            for tag, config in self.temp_config['mappings'].items():
                temp_channels.append(f"(@{config['channel']})")

            # Build flow channel list from config
            for tag, config in self.flow_config['mappings'].items():
                flow_channels.append(f"(@{config['channel']})")

            temp_list = ','.join(temp_channels)
            flow_list = ','.join(flow_channels)
            all_channels = f'{temp_list},{flow_list}' if temp_list and flow_list else temp_list or flow_list

            # Configure temperature channels for thermocouple measurement
            if temp_channels:
                sensor_type = self.temp_config.get('sensor_type', 'K')
                await asyncio.to_thread(self.instrument.write, f'CONF:TEMP TC,{sensor_type},{temp_list}')

            # Configure current channels for 4-20mA measurement
            if flow_channels:
                current_range = self.flow_config.get('range', 0.02)
                await asyncio.to_thread(self.instrument.write, f'CONF:CURR:DC {current_range},{flow_list}')

            # Set up scan list
            if all_channels:
                await asyncio.to_thread(self.instrument.write, f'ROUT:SCAN {all_channels}')

            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to DAQ970A at {self.ip_address}: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        if self.instrument:
            try:
                await asyncio.to_thread(self.instrument.close)
                self.connected = False
                self.logger.info("Disconnected from DAQ970A")
            except Exception as e:
                self.logger.error(f"Error disconnecting from DAQ970A: {e}")

    async def read_all_channels(self) -> Dict[str, float]:
        if not self.connected or not self.instrument:
            raise ConnectionError("DAQ970A not connected")

        try:
            # Initiate scan and read all channels
            await asyncio.to_thread(self.instrument.write, 'INIT')

            # Wait for scan to complete
            await asyncio.sleep(0.1)

            # Read data
            data_str = await asyncio.to_thread(self.instrument.query, 'FETC?')
            values = [float(x) for x in data_str.strip().split(',')]

            # Parse results into dictionary using configured mappings
            result = {}
            value_index = 0

            # Process temperature channels based on mapping order
            temp_tags = list(self.temp_config['mappings'].keys())
            for i, tag in enumerate(temp_tags):
                if value_index < len(values):
                    result[tag] = values[value_index]
                else:
                    result[tag] = float('nan')
                value_index += 1

            # Process flow channels based on mapping order
            flow_tags = list(self.flow_config['mappings'].keys())
            min_current = self.flow_config.get('min_current', 4)
            max_current = self.flow_config.get('max_current', 20)
            current_range = max_current - min_current

            for i, tag in enumerate(flow_tags):
                if value_index < len(values):
                    # Convert current to flow percentage
                    current_ma = values[value_index] * 1000  # Convert A to mA
                    flow_percent = ((current_ma - min_current) / current_range) * 100
                    # Apply scale factor if configured
                    scale = self.flow_config['mappings'][tag].get('scale', 1.0)
                    result[tag] = max(0, min(100, flow_percent * scale))
                else:
                    result[tag] = float('nan')
                value_index += 1

            return result

        except Exception as e:
            self.logger.error(f"Error reading DAQ970A channels: {e}")
            raise

    async def read_temperature_channels(self) -> Dict[str, float]:
        full_data = await self.read_all_channels()
        return {k: v for k, v in full_data.items() if k in self.temp_config['mappings']}

    async def read_flow_channels(self) -> Dict[str, float]:
        full_data = await self.read_all_channels()
        return {k: v for k, v in full_data.items() if k in self.flow_config['mappings']}

    def get_channel_info(self, tag: str) -> Optional[Dict]:
        """Get channel information for a given tag"""
        if tag in self.temp_config['mappings']:
            return self.temp_config['mappings'][tag]
        elif tag in self.flow_config['mappings']:
            return self.flow_config['mappings'][tag]
        return None

    def get_channel_coordinates(self, tag: str) -> Optional[Dict]:
        """Get spatial coordinates for a temperature probe"""
        if tag in self.temp_config['mappings']:
            return self.temp_config['mappings'][tag].get('coordinates', {'r': 0, 'z': 0, 'theta': 0})
        return None

    def get_all_coordinates(self) -> Dict[str, Dict]:
        """Get coordinates for all temperature probes"""
        coordinates = {}
        for tag, config in self.temp_config['mappings'].items():
            coordinates[tag] = config.get('coordinates', {'r': 0, 'z': 0, 'theta': 0})
        return coordinates

    def update_config(self, temp_config: Dict = None, flow_config: Dict = None):
        """Update channel configuration (requires reconnection to take effect)"""
        if temp_config:
            self.temp_config = temp_config
        if flow_config:
            self.flow_config = flow_config

    def is_connected(self) -> bool:
        return self.connected