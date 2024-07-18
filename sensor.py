import serial
import csv
import datetime
import time
import os
import sys
import glob
import oss2
from dotenv import load_dotenv
import RPi.GPIO as GPIO

# Load environment variables from .env file
load_dotenv()

# Fetch environment variables
access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
endpoint = os.getenv('OSS_ENDPOINT')

# Check if any environment variables are missing
if not all([access_key_id, access_key_secret, endpoint]):
    print("Error: One or more environment variables are missing.")
    sys.exit(1)

# Authenticate with OSS
auth = oss2.Auth(access_key_id, access_key_secret)
service = oss2.Service(auth, endpoint)

# Parameters
node_name = "default_node"  # Default node name if needed
data_buffer_time = 86400  # 24 hours in seconds

def upload_bucket(bucket_name, file_path):
    try:
        local_file_name = file_path.split('/')[-1]
        upload_file_path = file_path
        try:
            bucket = oss2.Bucket(auth, endpoint, bucket_name)
            bucket.create_bucket()
        except:
            pass
        # Upload the created file
        with open(upload_file_path, "rb") as data:
            bucket.put_object(local_file_name, data)
        print(datetime.datetime.now(), 'Finished uploading')
    except Exception as ex:
        print('Exception:')
        print(ex)

def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

def delete_file(filepath):
    os.remove(filepath)
    print('Deleted', filepath)

def is_valid_timestamp(ts):
    try:
        # Convert timestamp to datetime object
        dt = datetime.datetime.fromtimestamp(int(ts))
        # Check if datetime object is valid and the date is within a reasonable range
        if dt.year > 2000:
            return True
        else:
            return False
    except:
        return False

# Initialize GPIO and serial port detection
ports = []
last_ports = []
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
while True:
    ports = serial_ports()
    if len(last_ports) == 0:
        last_ports = serial_ports()
    com_port = list(set(ports).symmetric_difference(set(last_ports)))
    GPIO.output(17, GPIO.HIGH)
    if len(com_port) > 0:
        GPIO.output(17, GPIO.LOW)
        GPIO.cleanup()
        print("Worked")
        break
    print(ports, com_port)
    last_ports = ports

# Setup serial port
ser = serial.Serial(
    port=com_port[0],
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=0
)

print("connected to: " + ser.portstr)

# Setup data logging
epoch_time = int(round(time.time())) + 25200
folder_name = '/home/pi/Raw_data'
data_filename = f'{folder_name}/data_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv'

# Create the raw data folder if it doesn't exist
try:
    os.mkdir(folder_name)
    print("successful")
except OSError as error:
    print(error)

struct = ["Timestamp", "Temperature", "Humidity", "Pressure", "PM1.0", "PM2.5", "PM10"]

# Main data logging loop
while True:
    epoch_time = int(round(time.time())) + 25200  # offset for GMT+7
    # Read data from serial
    line = ser.readline()
    # Create new data file if file does not exist
    try:
        with open(data_filename, 'r', newline=''):
            pass
    except FileNotFoundError:
        with open(data_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(struct)

    # Read serial data and save to log file
    if len(line) > 0:
        data = line.decode('UTF-8').rstrip().split(',')
        if is_valid_timestamp(data[0]):  # Check if timestamp is valid
            data[0] = datetime.datetime.fromtimestamp(int(data[0])).strftime("%Y-%m-%d %H:%M:%S")
            print(data)
            with open(data_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        else:
            print("Invalid timestamp, data not recorded:")

    # Upload data file and delete old data file at the end of a day
    if (epoch_time) % 86400 == 0:
        upload_bucket('node-' + node_name, data_filename)
        # Delete old data file
        delete_file(data_filename)

ser.close()
