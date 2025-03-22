from ZMXService import ZMXSensorService

zmx = ZMXSensorService(ipAddress="192.168.8.50",scan_rate=0.5)

print(zmx.read_roi_data())