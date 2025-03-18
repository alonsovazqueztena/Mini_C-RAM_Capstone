import socket
import websocket
import logging

def get_host_ip():
    """Retrieve the local host computer's IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        logging.error(f"Error determining host IP: {e}")
        return "127.0.0.1"

class HeadlightController:
    """Controls a moving headlight via QLC+ WebSocket DMX commands."""
    
    def __init__(self, port=9999):
        self.QLC_IP = get_host_ip()
        self.QLC_WS_URL = f"ws://{self.QLC_IP}:{port}/qlcplusWS"
        self.ws = websocket.WebSocket()
        try:
            self.ws.connect(self.QLC_WS_URL)
            logging.info(f"Connected to QLC+ WebSocket at {self.QLC_WS_URL}.")
        except Exception as e:
            logging.error(f"Failed to connect to QLC+ WebSocket: {e}")
            raise
    
    def send_dmx_value(self, channel, value):
        """Send a DMX value to the specified channel through the WebSocket connection."""
        cmd = f"CH|{channel}|{int(value)}"
        try:
            self.ws.send(cmd)
            logging.info(f"Sent: {cmd}")
        except Exception as e:
            logging.error(f"Error sending DMX value: {e}")
    
    def close(self):
        self.ws.close()
        logging.info("WebSocket closed.")
