from pycomm3 import LogixDriver, CIPDriver
import time
import logging
import asyncio
import torch
from asyncua import Client

#print(asyncua.__version__)


SERVER_URL = "opc.tcp://192.168.8.157:4840"
#TAG_NODE_ID = "ns=4;s=|var|WAGO 751-9301 Compact Controller 100.Application.GVL.xDOs[0]"
TAG_NODE_ID = "ns=4;s=|var|WAGO 751-9301 Compact Controller 100.Application.GVL.iTemp"
TAG_NODE_ID_W = "ns=4;s=|var|WAGO 751-9301 Compact Controller 100.Application.GVL.xDOs[0]"
bit = False

async def connect_opcua():
    """Asynchronously connects to the OPC UA server."""
    client = Client(SERVER_URL)
    client.set_user("admin")
    client.set_password("cse")
    try:
        await client.connect()
        print(f"Connected to OPC UA Server at {SERVER_URL}")
        return client
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

async def read_opcua_value(client, node_id):
    """Reads and prints the value of a given OPC UA node asynchronously."""
    try:
        node = client.get_node(node_id)
        value = await node.read_value()
        print(f"Node {node_id} Value: {value}")
        #obs = torch.tensor([float(value[0]), float(value[1])], dtype=torch.float32, device='cpu')
        #print(obs, obs.size())
        return value
    except Exception as e:
        print(f"Error reading node {node_id}: {e}")

async def write_opcua_value(client, node_id, value):
    """Writes a value to a given OPC UA node asynchronously."""
    try:
        node = client.get_node(node_id)
        await node.write_value(value)
        print(f"Node {node_id} Updated to: {value}")
    except Exception as e:
        print(f"Error writing to node {node_id}: {e}")

async def main():
    """Main async function to connect and read values periodically."""
    client = await connect_opcua()
    
    if client:
        try:
            while True:
                await read_opcua_value(client, TAG_NODE_ID_W)
                await asyncio.sleep(2)  # Read every 2 seconds
                global bit
                bit = not bit
                await write_opcua_value(client, TAG_NODE_ID_W, value = not bit)
                await asyncio.sleep(2)  # Read every 2 seconds

        except KeyboardInterrupt:
            print("Disconnecting...")
        finally:
            await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
