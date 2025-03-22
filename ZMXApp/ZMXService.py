from pycomm3 import LogixDriver, CIPDriver
import time
import logging
from typing import Dict, Optional

class ZMXSensorService:

    def __init__(self, ipAddress : str = "192.168.8.50", scan_rate : float = 0.25):
        self.ipAddress = ipAddress
        self.scan_rate = max(scan_rate, 0.25)
        self.plc = None
        self.connected = False
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def connect(self) -> bool:
        try:
            self.plc = LogixDriver(self.ipAddress)
            self.plc.open()
            self.connected = True
            logging.info(f"Connected to ZMX Sensor at {self.ipAddress}")
            return True
        except Exception as e:
            logging.error(f"Connection error: {e}")
            return False
        
    def read_roi_data(self) -> Optional[Dict]:
        if not self.connect():
            return None
        
        try:
            data = self.plc.generic_message(
                service=0x0E,
                class_code=0x04,
                instance=100,
                attribute=3,
                request_data=b''
            )
            if data.error:
                logging.err(f"Read Error: {data.error}")
                return None
            roi_data = {
                'anchor_x' : data.value[53],
                'anchor_y' : data.value[54],
                'anchor_z' : data.value[55],
                'length_x' : data.value[56],
                'width_y'  : data.value[57],
                'height_z' : data.value[58],
                'items_counted' : data.value[26]
            }
            return roi_data
        except Exception as e:
            logging.error(f"Erro reading ROI data :{e}")

    def run_service(self):
        if not self.connected:
            return
        
        while self.connected:
            try:
                roi_data = self.read_roi_data()
                if roi_data:
                    logging.info(f"ROI data: {roi_data}")
                time.sleep(self.scan_rate)
            except Exception as e:
                logging.error(f"Service Error: {e}")
                self.connected = False
                #attempt to reconnect
                time.sleep(1)
                self.connect()

def main():
    service = ZMXSensorService()
    service.run_service()

if __name__ == "__main__":
    main()
