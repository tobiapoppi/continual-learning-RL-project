import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import time

# Define a simple feed-forward neural network

class SpaceInvadersNN(nn.Module):
    def __init__(self, output_dim):
        super(SpaceInvadersNN, self).__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),  # This might need adjustments based on the output shape of the conv layers
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return self.fc(x)


#class SimpleNN(nn.Module):
#    def __init__(self, input_dim, output_dim):
#        super(SimpleNN, self).__init__()
#        self.fc = nn.Sequential(
#            nn.Linear(input_dim, 128),
#            nn.ReLU(),
#            nn.Linear(128, output_dim)
#        )
#
#    def forward(self, x):
#        return self.fc(x)

# Define the EWC loss
class EWCLoss(nn.Module):
    def __init__(self):
        super(EWCLoss, self).__init__()

    def forward(self, model, fisher, opt_params):
        ewc_loss = 0
        for param, opt_param, fish in zip(model.parameters(), opt_params, fisher):
            ewc_loss += (fish * (param - opt_param) ** 2).sum()
        return ewc_loss

# Train the model on a task and compute Fisher Information
def train_and_compute_fisher(model, optimizer, env, epochs=10, fisher_samples=1000):
    criterion = nn.MSELoss()
    for _ in range(epochs):
        state = env.reset()
        done = False
        while not done:
            if (len(state)==2):
                state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0)
            else:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            #print(state.shape)
            action = model(state)[0].argmax().item()
            next_state, reward, done, _, _ = env.step(action)
            #aa = env.step(action)  
            target = torch.tensor(reward + (0.99 * model(torch.tensor(next_state, dtype=torch.float32)).max().item() * (not done)), dtype=torch.float32)
            #pred = model(torch.tensor(state[0], dtype=torch.float32).unsqueeze(0))[action]
            pred = model(torch.tensor(state, dtype=torch.float32))[0][action]
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state

    fisher_info = []
    for param in model.parameters():
        fisher_info.append(torch.zeros_like(param))

    model.eval()
    for _ in range(fisher_samples):
        state = env.reset()
        done = False
        while not done:
            if (len(state)==2):
                state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0)
            else:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = nn.Softmax(dim=-1)(model(state))
            action = torch.multinomial(probs, 1).item()
            next_state, reward, done, _, _ = env.step(action)
            model.zero_grad()
            probs[0][action].backward(retain_graph=True)
            for param, fish in zip(model.parameters(), fisher_info):
                fish += param.grad ** 2 / fisher_samples
            state = next_state

    return [param.data.clone() for param in model.parameters()], fisher_info

# Train the model using EWC
def train_with_ewc(model, optimizer, env, prev_params, fisher, epochs=10, ewc_lambda=4000):
    criterion = nn.MSELoss()
    ewc_criterion = EWCLoss()
    for _ in range(epochs):
        state = env.reset()
        done = False
        while not done:
            if (len(state)==2):
                state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0)
            else:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = model(state)[0].argmax().item()
            next_state, reward, done, _, _ = env.step(action)
            target = torch.tensor(reward + (0.99 * model(torch.tensor(next_state, dtype=torch.float32)).max().item() * (not done)), dtype=torch.float32)
            pred = model(state)[0][action]
            loss = criterion(pred, target)
            ewc_loss = ewc_criterion(model, fisher, prev_params)
            total_loss = loss + ewc_lambda * ewc_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            state = next_state

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
model = SpaceInvadersNN(env.action_space.n)

#model = SimpleNN(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train on Task 1 and compute Fisher
opt_params, fisher = train_and_compute_fisher(model, optimizer, env)

# Mock a Task 2 by simply re-training on the same environment (for simplicity)
# In a real scenario, you'd use a different environment or modify the current one
train_with_ewc(model, optimizer, env, opt_params, fisher)


#### TEST
def test_model(model, env, episodes=100):
    scores = []
    for _ in range(episodes):
        time.sleep(0.5)
        state = env.reset()
        done = False
        score = 0
        while not done:
            if (len(state)==2):
                state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0)
            else:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            env.render()
            action = model(state)[0].argmax().item()
            action = model(state)[0].argmax().item()
            next_state, reward, done, _, _ = env.step(action)
            score += 1
            state = next_state
        scores.append(score)
    return np.mean(scores)

# Mock testing since we don't have gym here
# Uncomment and run the following line in your environment after training
# average_score = test_model(model, env)
# print(f"Average score over {episodes} episodes: {average_score}")


media = test_model(model, env)
media