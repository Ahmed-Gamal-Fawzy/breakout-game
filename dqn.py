import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gameTRY import Breakout
import sys
import time




class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.05
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 750000
        self.minibatch_size = 32
        self.explore = 3000000

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        return self.fc5(x)


def preprocessing(image):
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (1, 84, 84)).astype(np.float32)
    return torch.from_numpy(image_data)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if m.weight is not None:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def train(model, start):
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    game_state = Breakout()
    D = deque(maxlen=model.replay_memory_size)

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    image_data, _, _ = game_state.take_action(action)
    state = torch.cat([preprocessing(image_data)] * 4).unsqueeze(0)
    epsilon = model.initial_epsilon
    iteration = 0

    while iteration < model.number_of_iterations:
        output = model(state)[0]
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        action_index = torch.randint(model.number_of_actions, ()) if random.random() <= epsilon else torch.argmax(output)
        action[action_index] = 1
        
        if epsilon > model.final_epsilon:
            epsilon -= (model.initial_epsilon - model.final_epsilon) / model.explore

        image_data_1, reward, terminal = game_state.take_action(action)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], preprocessing(image_data_1))).unsqueeze(0)
        
        D.append((state, action.unsqueeze(0), torch.tensor([[reward]], dtype=torch.float32), state_1, terminal))
        
        if len(D) >= model.minibatch_size:
            minibatch = random.sample(D, model.minibatch_size)
            state_batch = torch.cat([d[0] for d in minibatch])
            action_batch = torch.cat([d[1] for d in minibatch])
            reward_batch = torch.cat([d[2] for d in minibatch])
            state_1_batch = torch.cat([d[3] for d in minibatch])
            
            output_1_batch = model(state_1_batch).detach()
            y_batch = reward_batch + model.gamma * torch.max(output_1_batch, dim=1, keepdim=True)[0] * ~torch.tensor([d[4] for d in minibatch], dtype=torch.bool).unsqueeze(1)
            
            q_value = torch.sum(model(state_batch) * action_batch, dim=1, keepdim=True)
            optimizer.zero_grad()
            loss = criterion(q_value, y_batch)
            loss.backward()
            optimizer.step()

        state = state_1
        iteration += 1

        if iteration % 10000 == 0:
            torch.save(model, f"trained_model/current_model_{iteration}.pth")

        print(f"Iteration: {iteration}, Time: {time.time() - start:.2f}s, Epsilon: {epsilon:.5f}, Action: {action_index}, Reward: {reward}")


def test(model):
    game_state = Breakout()
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, _, _ = game_state.take_action(action)
    state = torch.cat([preprocessing(image_data)] * 4).unsqueeze(0)

    while True:
        output = model(state)[0]
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        action[torch.argmax(output)] = 1
        image_data_1, _, _ = game_state.take_action(action)
        state = torch.cat((state.squeeze(0)[1:, :, :], preprocessing(image_data_1))).unsqueeze(0)


def main(mode=None):
    os.makedirs('trained_model', exist_ok=True)
    model_path = 'trained_model/current_model_420000.pth'

    if mode == 'test':
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location='cpu').eval()
            test(model)
        else:
            print(f"Error: Model file '{model_path}' not found!")
    elif mode == 'train':
        model = NeuralNetwork()
        model.apply(init_weights)
        train(model, time.time())
    elif mode == 'continue':
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location='cpu').eval()
            train(model, time.time())
        else:
            print(f"Error: Model file '{model_path}' not found!")
    else:
        print("Usage: python dqn.py [train|test|continue]")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'train')