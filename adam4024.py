import asyncio
import logging
from typing import Dict, List, Optional
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException

class ADAM4024Channel:
    def __init__(self, channel_id: int, name: str = None, valve_type: str = "Control Valve"):
        self.channel_id = channel_id
        self.name = name or f"Valve_{channel_id}"
        self.valve_type = valve_type
        self.output_voltage = 0.0  # 0-10V
        self.output_percent = 0.0  # 0-100%
        self.enabled = True
        self.min_voltage = 0.0
        self.max_voltage = 10.0

    def set_voltage(self, voltage: float):
        """Set output voltage (0-10V)"""
        self.output_voltage = max(self.min_voltage, min(self.max_voltage, voltage))
        self.output_percent = (self.output_voltage / self.max_voltage) * 100

    def set_percent(self, percent: float):
        """Set output as percentage (0-100%)"""
        self.output_percent = max(0, min(100, percent))
        self.output_voltage = (self.output_percent / 100) * self.max_voltage

    def get_dac_value(self) -> int:
        """Convert voltage to DAC value for ADAM-4024 (0-4095 for 0-10V)"""
        # ADAM-4024 uses 12-bit DAC: 4095 = 10V, 0 = 0V
        return int((self.output_voltage / 10.0) * 4095)

    def __repr__(self):
        return f"ADAM4024Channel(id={self.channel_id}, name='{self.name}', voltage={self.output_voltage:.2f}V, percent={self.output_percent:.1f}%)"

class ADAM4024Controller:
    def __init__(self, unit_id: int = 1, client: ModbusSerialClient = None):
        self.unit_id = unit_id
        self.client = client
        self.connected = False
        self.logger = logging.getLogger(__name__)

        # Initialize 4 analog output channels
        self.channels: Dict[int, ADAM4024Channel] = {}
        for i in range(4):
            channel_id = i + 1
            self.channels[channel_id] = ADAM4024Channel(
                channel_id=channel_id,
                name=f"Valve {channel_id}",
                valve_type="Control Valve"
            )

        # ADAM-4024 Modbus register mapping
        self.registers = {
            'analog_output_ch1': 0x0000,  # Analog Output Channel 1
            'analog_output_ch2': 0x0001,  # Analog Output Channel 2
            'analog_output_ch3': 0x0002,  # Analog Output Channel 3
            'analog_output_ch4': 0x0003,  # Analog Output Channel 4
            'status_register': 0x0040,    # Status register
            'configuration': 0x0041,      # Configuration register
        }

    def set_modbus_client(self, client: ModbusSerialClient):
        """Set the Modbus client (shared with VLT drives)"""
        self.client = client
        if client and hasattr(client, 'is_socket_open') and client.is_socket_open():
            self.connected = True

    async def initialize_device(self) -> bool:
        """Initialize ADAM-4024 device configuration"""
        if not self.client:
            self.logger.error("No Modbus client available")
            return False

        try:
            # Read device status to verify communication
            status_result = await asyncio.to_thread(
                self.client.read_holding_registers,
                address=self.registers['status_register'],
                count=1,
                unit=self.unit_id
            )

            if status_result.isError():
                self.logger.error(f"Failed to read ADAM-4024 status: {status_result}")
                return False

            # Configure device for 0-10V output mode if needed
            # This depends on the specific ADAM-4024 model and configuration
            config_result = await asyncio.to_thread(
                self.client.write_register,
                address=self.registers['configuration'],
                value=0x0000,  # Configure for 0-10V mode (model-specific)
                unit=self.unit_id
            )

            if config_result.isError():
                self.logger.warning(f"Configuration write failed: {config_result}")

            self.connected = True
            self.logger.info(f"ADAM-4024 unit {self.unit_id} initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing ADAM-4024 unit {self.unit_id}: {e}")
            return False

    async def set_channel_voltage(self, channel: int, voltage: float) -> bool:
        """Set analog output voltage for a specific channel"""
        if not self.client or not self.connected:
            self.logger.error("ADAM-4024 not connected")
            return False

        if channel not in self.channels:
            self.logger.error(f"Invalid channel: {channel}")
            return False

        try:
            # Update channel object
            self.channels[channel].set_voltage(voltage)
            dac_value = self.channels[channel].get_dac_value()

            # Write to Modbus register
            register_address = self.registers[f'analog_output_ch{channel}']
            result = await asyncio.to_thread(
                self.client.write_register,
                address=register_address,
                value=dac_value,
                unit=self.unit_id
            )

            if result.isError():
                self.logger.error(f"Failed to set channel {channel} voltage: {result}")
                return False

            self.logger.info(f"ADAM-4024 Ch{channel}: Set to {voltage:.2f}V ({dac_value} DAC)")
            return True

        except Exception as e:
            self.logger.error(f"Error setting ADAM-4024 channel {channel} voltage: {e}")
            return False

    async def set_channel_percent(self, channel: int, percent: float) -> bool:
        """Set analog output as percentage (0-100%) for a specific channel"""
        if channel not in self.channels:
            self.logger.error(f"Invalid channel: {channel}")
            return False

        # Convert percentage to voltage
        voltage = (percent / 100.0) * self.channels[channel].max_voltage
        return await self.set_channel_voltage(channel, voltage)

    async def get_channel_status(self, channel: int) -> Optional[ADAM4024Channel]:
        """Get current status of a channel"""
        if channel in self.channels:
            return self.channels[channel]
        return None

    async def get_all_channels(self) -> Dict[int, ADAM4024Channel]:
        """Get status of all channels"""
        return self.channels.copy()

    async def set_all_channels_off(self):
        """Set all channels to 0V"""
        tasks = []
        for channel_id in self.channels.keys():
            tasks.append(self.set_channel_voltage(channel_id, 0.0))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(isinstance(r, bool) and r for r in results)

    async def emergency_stop(self):
        """Emergency stop - set all outputs to 0V immediately"""
        self.logger.warning("ADAM-4024 Emergency Stop - Setting all outputs to 0V")
        await self.set_all_channels_off()

    def configure_channel(self, channel: int, name: str, valve_type: str = "Control Valve",
                         min_voltage: float = 0.0, max_voltage: float = 10.0):
        """Configure channel parameters"""
        if channel in self.channels:
            self.channels[channel].name = name
            self.channels[channel].valve_type = valve_type
            self.channels[channel].min_voltage = min_voltage
            self.channels[channel].max_voltage = max_voltage
            self.logger.info(f"Configured ADAM-4024 Ch{channel}: {name} ({valve_type})")

    def is_connected(self) -> bool:
        """Check if device is connected"""
        return self.connected and self.client is not None

    async def read_channel_feedback(self, channel: int) -> Optional[float]:
        """Read actual output voltage from device (if supported by model)"""
        # Note: Not all ADAM-4024 models support readback
        # This is a placeholder for models that do support it
        try:
            if not self.client or not self.connected:
                return None

            # Some ADAM-4024 models have readback registers at different addresses
            readback_register = self.registers[f'analog_output_ch{channel}']

            result = await asyncio.to_thread(
                self.client.read_holding_registers,
                address=readback_register,
                count=1,
                unit=self.unit_id
            )

            if not result.isError():
                dac_value = result.registers[0]
                voltage = (dac_value / 4095.0) * 10.0  # Convert DAC to voltage
                return voltage

        except Exception as e:
            self.logger.debug(f"Channel {channel} readback not available: {e}")

        return None

    def get_device_info(self) -> Dict[str, any]:
        """Get device information"""
        return {
            'model': 'ADAM-4024',
            'unit_id': self.unit_id,
            'channels': len(self.channels),
            'connected': self.connected,
            'output_range': '0-10V',
            'resolution': '12-bit (4095 steps)',
            'channel_names': [ch.name for ch in self.channels.values()]
        }