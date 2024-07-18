import serial

# Replace 'COM3' with the actual COM port number for your ST-Link VCP
ser = serial.Serial('/dev/ttyS0', 115200, timeout=1)

while True:
    line = ser.readline().decode('utf-8').strip()
    print(line)
