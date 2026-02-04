
import asyncio
import struct
from bleak import BleakClient, BleakScanner
import numpy as np
from datetime import datetime
from collections import deque

# Polar H10 BLE Service UUIDs
HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

class PolarH10:
    def __init__(self, device_name="Polar H10"):
        self.device_name = device_name
        self.device = None
        self.client = None
        self.rr_intervals = deque(maxlen=1000)  # Store last 1000 RR intervals
        self.heart_rates = deque(maxlen=100)
        
    async def find_device(self):
        """Scan for Polar H10 device"""
        print(f"Scanning for {self.device_name}...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        for device in devices:
            if device.name and self.device_name in device.name:
                print(f"Found device: {device.name} ({device.address})")
                self.device = device
                return device
        
        raise Exception(f"Could not find {self.device_name}")
    
    def parse_heart_rate_data(self, sender, data):
        """
        Parse heart rate measurement data according to BLE Heart Rate Service spec
        
        Byte 0: Flags
        - Bit 0: Heart Rate Value Format (0 = uint8, 1 = uint16)
        - Bit 1-2: Sensor Contact Status
        - Bit 3: Energy Expended Status
        - Bit 4: RR-Interval present
        
        Following bytes: HR value, then RR intervals if present
        """
        flags = data[0]
        hr_format = flags & 0x01  # 0 = uint8, 1 = uint16
        rr_present = (flags & 0x10) != 0
        
        # Parse heart rate
        if hr_format == 0:
            heart_rate = data[1]
            offset = 2
        else:
            heart_rate = struct.unpack('<H', data[1:3])[0]
            offset = 3
        
        self.heart_rates.append(heart_rate)
        
        # Parse RR intervals if present
        if rr_present:
            # RR intervals are in units of 1/1024 seconds
            rr_data = data[offset:]
            num_rr = len(rr_data) // 2
            
            for i in range(num_rr):
                rr_raw = struct.unpack('<H', rr_data[i*2:(i+1)*2])[0]
                rr_ms = (rr_raw / 1024.0) * 1000.0  # Convert to milliseconds
                self.rr_intervals.append(rr_ms)
                
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] HR: {heart_rate} bpm | RR intervals: {[f'{rr:.1f}ms' for rr in list(self.rr_intervals)[-num_rr:]]}")
    
    def calculate_hrv_metrics(self):
        """Calculate common HRV metrics from RR intervals"""
        if len(self.rr_intervals) < 2:
            return None
        
        rr_array = np.array(self.rr_intervals)
        
        # Time domain metrics
        mean_rr = np.mean(rr_array)
        sdnn = np.std(rr_array, ddof=1)  # Standard deviation of NN intervals
        
        # Calculate successive differences
        diff_rr = np.diff(rr_array)
        rmssd = np.sqrt(np.mean(diff_rr**2))  # Root mean square of successive differences
        
        # NN50: number of successive differences > 50ms
        nn50 = np.sum(np.abs(diff_rr) > 50)
        pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
        
        metrics = {
            'mean_rr': mean_rr,
            'mean_hr': 60000 / mean_rr if mean_rr > 0 else 0,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'nn50': nn50,
            'pnn50': pnn50,
            'num_intervals': len(rr_array)
        }
        
        return metrics
    
    async def connect_and_stream(self, duration=60):
        """Connect to device and stream HRV data"""
        if not self.device:
            await self.find_device()
        
        print(f"\nConnecting to {self.device.name}...")
        
        async with BleakClient(self.device.address) as client:
            self.client = client
            print(f"Connected: {client.is_connected}")
            
            # Subscribe to heart rate notifications
            await client.start_notify(HEART_RATE_MEASUREMENT_UUID, self.parse_heart_rate_data)
            print(f"Streaming HRV data for {duration} seconds...\n")
            
            # Stream for specified duration
            await asyncio.sleep(duration)
            
            # Stop notifications
            await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)
            print("\n" + "="*70)
            print("Data collection complete!")
            
            # Calculate and display HRV metrics
            metrics = self.calculate_hrv_metrics()
            if metrics:
                print("\n--- HRV Metrics ---")
                print(f"Number of RR intervals: {metrics['num_intervals']}")
                print(f"Mean RR interval: {metrics['mean_rr']:.1f} ms")
                print(f"Mean Heart Rate: {metrics['mean_hr']:.1f} bpm")
                print(f"SDNN (standard deviation): {metrics['sdnn']:.1f} ms")
                print(f"RMSSD: {metrics['rmssd']:.1f} ms")
                print(f"NN50: {metrics['nn50']}")
                print(f"pNN50: {metrics['pnn50']:.1f}%")
                print("="*70)
            
            return metrics


async def main():
    """Main function to run the HRV data collection"""
    # You can customize the device name if your Polar H10 has a different identifier
    polar = PolarH10(device_name="Polar H10")
    
    try:
        # Stream data for 60 seconds (adjust as needed)
        metrics = await polar.connect_and_stream(duration=60)
        
        # Access the collected RR intervals if needed for further analysis
        if metrics:
            print(f"\nCollected {len(polar.rr_intervals)} RR intervals")
            print(f"RR intervals available in polar.rr_intervals")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
