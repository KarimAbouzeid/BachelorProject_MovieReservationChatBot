import matplotlib.pyplot as plt

# Time data
CS_data = {
    'Episode 0': 2,
    'Episode 1': 8,
    'Episode 2': 9,
    'Episode 3': 11,
    'Episode 4': 13,
    'Episode 5': 16,
    'Episode 6': 18,
    'Episode 7': 20,
    'Episode 8': 23,
    'Episode 9': 26,
    'Episode 10': 28,
    'Episode 11': 29,
    'Episode 12': 34,
    'Episode 13': 35,
    'Episode 14': 39,
    'Episode 15': 40,
    'Episode 16': 42,
    'Episode 17': 44,
    'Episode 18': 49,
    'Episode 19': 51,
    'Episode 20': 53,
    'Episode 21': 57,
    'Episode 22': 56,
    'Episode 23': 58,
    'Episode 24': 59,
    'Episode 25': 61,
    'Episode 26': 63,
    'Episode 27': 64,
    'Episode 28': 66,
    'Episode 29': 67,
    'Episode 30': 69,
    'Episode 31': 70,
    'Episode 32': 72,
    'Episode 33': 73,
    'Episode 34': 75,
    'Episode 35': 77,
    'Episode 36': 79,
    'Episode 37': 80,
    'Episode 38': 81,
    'Episode 39': 83,
    'Episode 40': 84,
    'Episode 41': 86,
    'Episode 42': 87,
    'Episode 43': 88,
    'Episode 44': 90,
    'Episode 45': 91,
    'Episode 46': 92,
    'Episode 47': 94,
    'Episode 48': 95,
    'Episode 49': 96
}

O_data = {
    'Episode 0': 0,
    'Episode 1': 0,
    'Episode 2': 0,
    'Episode 3': 1,
    'Episode 4': 3,
    'Episode 5': 5,
    'Episode 6': 5,
    'Episode 7': 5,
    'Episode 8': 6,
    'Episode 9': 7,
    'Episode 10': 7,
    'Episode 11': 7,
    'Episode 12': 8,
    'Episode 13': 9,
    'Episode 14': 10,
    'Episode 15': 11,
    'Episode 16': 11,
    'Episode 17': 11,
    'Episode 18': 11,
    'Episode 19': 11,
    'Episode 20': 12,
    'Episode 21': 12,
    'Episode 22': 13,
    'Episode 23': 13,
    'Episode 24': 14,
    'Episode 25': 14,
    'Episode 26': 14,
    'Episode 27': 15,
    'Episode 28': 15,
    'Episode 29': 16,
    'Episode 30': 16,
    'Episode 31': 16,
    'Episode 32': 17,
    'Episode 33': 17,
    'Episode 34': 22,
    'Episode 35': 22,
    'Episode 36': 23,
    'Episode 37': 23,
    'Episode 38': 23,
    'Episode 39': 24,
    'Episode 40': 24,
    'Episode 41': 24,
    'Episode 42': 24,
    'Episode 43': 25,
    'Episode 44': 25,
    'Episode 45': 26,
    'Episode 46': 26,
    'Episode 47': 27,
    'Episode 48': 27,
    'Episode 49': 27
}


# Extract episode numbers and times
episode_numbers = range(50)
episode1_times = [CS_data.get(f"Episode {episode}", 0) for episode in episode_numbers]
episode2_times = [O_data.get(f"Episode {episode}", 0) for episode in episode_numbers]


# Create the graph
# plt.plot(episode_numbers, episode_times)
# plt.xlabel('Number of Episodes')
# plt.ylabel('Time (minutes)')
# plt.title('Episode Time')
# plt.grid(True)
# plt.show()

plt.plot(episode_numbers, episode1_times, label='Client/Server')
plt.plot(episode_numbers, episode2_times, linestyle='--', label='Original')

# plt.plot(episode1_times, label='Time 1')
# plt.plot(episode2_times, label='Time 2')

plt.legend()

plt.xlabel('Number of Epochs')
plt.ylabel('Time (minutes)')
plt.title('Training time without NLU and NLG')


plt.show()