from opcuaService import *
import asyncio
import torch
import numpy as np
from stable_baselines3 import PPO


obsRaw : list
actionRaw : list

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[512, 512],
                     squash_output=False)
# Create the agent
#model = PPO("MlpPolicy", policy_kwargs=policy_kwargs, verbose=1)
model = PPO.load("model_52928000_steps.zip")


async def main():
    """Main async function to connect and read values periodically."""
    client = await connect_opcua()
    if client:
        try:
            while True:
                obsRaw =  await read_opcua_value(client, TAG_NODE_ID)
                await asyncio.sleep(0.25)  # Read every 0.25 seconds
                obs = (torch.tensor(obsRaw[0:8],dtype=torch.float32,device='cpu') / 100.0).unsqueeze(0)
                print(obs)
                with torch.no_grad():
                    #action, _ = model.predict(obs)
                    action_dist = model.policy.get_distribution(obs)
                    action = action_dist.get_actions()
                    
                action = action.flatten()
                
                print(action)
                print(action[action > 20])
                
                # print(torch.mean(action[0:16]), torch.mean(action[16:32]), torch.mean(action[32:48]), torch.mean(action[48:64]),
                #        torch.mean(action[64:80]), torch.mean(action[80:96]), torch.mean(action[96:112]), torch.mean(action[112:128]))


        except KeyboardInterrupt:
            print("Disconnecting...")
        finally:
            await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())