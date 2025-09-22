import sqlite3
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os

class DataRecorder:
    def __init__(self, db_path: str = "data_recordings.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.recording = False
        self.current_test_id = None
        self.recording_frequency = 1.0  # seconds
        self.recording_task = None

        # Ensure database directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        # Initialize database
        self.init_database()

    def init_database(self):
        """Initialize the SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tests table to store test metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL UNIQUE,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'active',
                    description TEXT,
                    recording_frequency REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create data_points table to store all recorded data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data_json TEXT NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES tests (id)
                )
            ''')

            # Create temperature_data table for structured temperature data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS temperature_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    channel_tag TEXT NOT NULL,
                    temperature REAL,
                    units TEXT DEFAULT '°C',
                    FOREIGN KEY (test_id) REFERENCES tests (id)
                )
            ''')

            # Create flow_data table for structured flow data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS flow_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    channel_tag TEXT NOT NULL,
                    flow_percent REAL,
                    current_ma REAL,
                    FOREIGN KEY (test_id) REFERENCES tests (id)
                )
            ''')

            # Create drive_data table for VLT drive data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drive_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    drive_id INTEGER NOT NULL,
                    speed_setpoint REAL,
                    actual_speed REAL,
                    status TEXT,
                    FOREIGN KEY (test_id) REFERENCES tests (id)
                )
            ''')

            # Create valve_data table for ADAM-4024 valve data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS valve_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    valve_id INTEGER NOT NULL,
                    voltage REAL,
                    percent REAL,
                    valve_name TEXT,
                    FOREIGN KEY (test_id) REFERENCES tests (id)
                )
            ''')

            # Create control_loop_data table for PID control loop data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS control_loop_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    loop_id TEXT NOT NULL,
                    control_type TEXT NOT NULL,
                    setpoint REAL,
                    process_value REAL,
                    output REAL,
                    error REAL,
                    enabled BOOLEAN,
                    FOREIGN KEY (test_id) REFERENCES tests (id)
                )
            ''')

            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_points_test_timestamp ON data_points (test_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_temperature_test_timestamp ON temperature_data (test_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_flow_test_timestamp ON flow_data (test_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drive_test_timestamp ON drive_data (test_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_valve_test_timestamp ON valve_data (test_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_control_loop_test_timestamp ON control_loop_data (test_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_control_loop_id ON control_loop_data (test_id, loop_id, timestamp)')

            conn.commit()
            conn.close()

            self.logger.info(f"Database initialized successfully at {self.db_path}")

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    def start_test(self, test_name: str, description: str = None, recording_frequency: float = 1.0) -> bool:
        """Start a new test recording session"""
        try:
            if self.recording:
                self.logger.warning("Recording already in progress")
                return False

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if test name already exists
            cursor.execute('SELECT id FROM tests WHERE test_name = ?', (test_name,))
            if cursor.fetchone():
                self.logger.error(f"Test name '{test_name}' already exists")
                conn.close()
                return False

            # Create new test record
            start_time = datetime.now()
            cursor.execute('''
                INSERT INTO tests (test_name, start_time, status, description, recording_frequency)
                VALUES (?, ?, 'active', ?, ?)
            ''', (test_name, start_time, description, recording_frequency))

            self.current_test_id = cursor.lastrowid
            self.recording_frequency = recording_frequency
            self.recording = True

            conn.commit()
            conn.close()

            self.logger.info(f"Started test recording: {test_name} (ID: {self.current_test_id})")
            return True

        except Exception as e:
            self.logger.error(f"Error starting test: {e}")
            return False

    def stop_test(self) -> bool:
        """Stop the current test recording session"""
        try:
            if not self.recording or not self.current_test_id:
                self.logger.warning("No recording in progress")
                return False

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Update test record with end time
            end_time = datetime.now()
            cursor.execute('''
                UPDATE tests SET end_time = ?, status = 'completed'
                WHERE id = ?
            ''', (end_time, self.current_test_id))

            conn.commit()
            conn.close()

            self.logger.info(f"Stopped test recording (ID: {self.current_test_id})")

            self.recording = False
            self.current_test_id = None

            return True

        except Exception as e:
            self.logger.error(f"Error stopping test: {e}")
            return False

    def record_data_point(self, data_point: Dict[str, Any]) -> bool:
        """Record a single data point to the database"""
        if not self.recording or not self.current_test_id:
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = data_point.get('timestamp', datetime.now())

            # Store complete data point as JSON
            cursor.execute('''
                INSERT INTO data_points (test_id, timestamp, data_json)
                VALUES (?, ?, ?)
            ''', (self.current_test_id, timestamp, json.dumps(data_point, default=str)))

            # Store structured temperature data
            for key, value in data_point.items():
                if key.startswith('T') and len(key) <= 4 and isinstance(value, (int, float)):
                    cursor.execute('''
                        INSERT INTO temperature_data (test_id, timestamp, channel_tag, temperature)
                        VALUES (?, ?, ?, ?)
                    ''', (self.current_test_id, timestamp, key, value))

                # Store structured flow data
                elif key.startswith('F') and len(key) <= 4 and isinstance(value, (int, float)):
                    cursor.execute('''
                        INSERT INTO flow_data (test_id, timestamp, channel_tag, flow_percent)
                        VALUES (?, ?, ?, ?)
                    ''', (self.current_test_id, timestamp, key, value))

            # Store drive data
            for key, value in data_point.items():
                if key.startswith('drive_') and isinstance(value, (int, float)):
                    parts = key.split('_')
                    if len(parts) >= 3:
                        drive_id = int(parts[1])
                        param = '_'.join(parts[2:])

                        if param == 'speed':
                            # Check if record exists for this drive/timestamp
                            cursor.execute('''
                                SELECT id FROM drive_data
                                WHERE test_id = ? AND timestamp = ? AND drive_id = ?
                            ''', (self.current_test_id, timestamp, drive_id))

                            if cursor.fetchone():
                                cursor.execute('''
                                    UPDATE drive_data SET actual_speed = ?
                                    WHERE test_id = ? AND timestamp = ? AND drive_id = ?
                                ''', (value, self.current_test_id, timestamp, drive_id))
                            else:
                                cursor.execute('''
                                    INSERT INTO drive_data (test_id, timestamp, drive_id, actual_speed)
                                    VALUES (?, ?, ?, ?)
                                ''', (self.current_test_id, timestamp, drive_id, value))

                        elif param == 'setpoint':
                            cursor.execute('''
                                INSERT OR REPLACE INTO drive_data
                                (test_id, timestamp, drive_id, speed_setpoint)
                                VALUES (?, ?, ?, ?)
                                ON CONFLICT(test_id, timestamp, drive_id) DO UPDATE SET
                                speed_setpoint = excluded.speed_setpoint
                            ''', (self.current_test_id, timestamp, drive_id, value))

            # Store valve data
            for key, value in data_point.items():
                if key.startswith('valve_') and isinstance(value, (int, float, str)):
                    parts = key.split('_')
                    if len(parts) >= 3:
                        valve_id = int(parts[1])
                        param = '_'.join(parts[2:])

                        if param == 'voltage':
                            cursor.execute('''
                                INSERT OR IGNORE INTO valve_data (test_id, timestamp, valve_id, voltage)
                                VALUES (?, ?, ?, ?)
                            ''', (self.current_test_id, timestamp, valve_id, value))
                        elif param == 'percent':
                            cursor.execute('''
                                UPDATE valve_data SET percent = ?
                                WHERE test_id = ? AND timestamp = ? AND valve_id = ?
                            ''', (value, self.current_test_id, timestamp, valve_id))
                        elif param == 'name':
                            cursor.execute('''
                                UPDATE valve_data SET valve_name = ?
                                WHERE test_id = ? AND timestamp = ? AND valve_id = ?
                            ''', (value, self.current_test_id, timestamp, valve_id))

            # Store control loop data
            control_loops = {}
            for key, value in data_point.items():
                if key.startswith('control_') and '_' in key:
                    parts = key.split('_')
                    if len(parts) >= 3:
                        loop_id = parts[1]
                        param = '_'.join(parts[2:])

                        if loop_id not in control_loops:
                            control_loops[loop_id] = {}
                        control_loops[loop_id][param] = value

            # Insert control loop records
            for loop_id, loop_data in control_loops.items():
                if all(param in loop_data for param in ['setpoint', 'output', 'error', 'enabled']):
                    control_type = loop_data.get('type', 'temperature')
                    process_value = loop_data.get('temperature') or loop_data.get('flow')

                    cursor.execute('''
                        INSERT INTO control_loop_data
                        (test_id, timestamp, loop_id, control_type, setpoint, process_value, output, error, enabled)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        self.current_test_id, timestamp, loop_id, control_type,
                        loop_data['setpoint'], process_value, loop_data['output'],
                        loop_data['error'], loop_data['enabled']
                    ))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            self.logger.error(f"Error recording data point: {e}")
            return False

    def get_test_list(self) -> List[Dict[str, Any]]:
        """Get list of all tests"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, test_name, start_time, end_time, status, description, recording_frequency
                FROM tests ORDER BY start_time DESC
            ''')

            tests = []
            for row in cursor.fetchall():
                tests.append({
                    'id': row[0],
                    'test_name': row[1],
                    'start_time': row[2],
                    'end_time': row[3],
                    'status': row[4],
                    'description': row[5],
                    'recording_frequency': row[6]
                })

            conn.close()
            return tests

        except Exception as e:
            self.logger.error(f"Error getting test list: {e}")
            return []

    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.recording

    def get_current_test_name(self) -> Optional[str]:
        """Get the name of the currently recording test"""
        if not self.recording or not self.current_test_id:
            return None

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT test_name FROM tests WHERE id = ?', (self.current_test_id,))
            result = cursor.fetchone()

            conn.close()

            return result[0] if result else None

        except Exception as e:
            self.logger.error(f"Error getting current test name: {e}")
            return None

    def set_recording_frequency(self, frequency: float):
        """Set the recording frequency in seconds"""
        self.recording_frequency = frequency

    def get_recording_frequency(self) -> float:
        """Get the current recording frequency"""
        return self.recording_frequency