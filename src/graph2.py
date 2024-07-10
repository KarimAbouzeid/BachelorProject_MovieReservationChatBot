import matplotlib.pyplot as plt


CS_data = {
    0: 7,
    1: 16,
    2: 28,
    3: 37,
    4: 44,
    5: 53,
    6: 58,
    7: 64,
    8: 70,
    9: 79,
    10: 85,
    11: 93,
    12: 128,
    13: 135,
    14: 146,
    15: 155,
    16: 163,
    17: 171,
    18: 177,
    19: 185,
    20: 221
    }

O_data = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 5,
    5: 6,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 9,
    11: 11,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 17,
    18: 17,
    19: 18,
    20: 19
}


plt.plot(CS_data.keys(), list(CS_data.values()), label='Client/Server')
plt.plot(O_data.keys(), O_data.values(), linestyle='--', label='Original')

plt.legend()

plt.ylim(0, 225)
plt.xlabel('Number of Epochs')
plt.ylabel('Time (minutes)')
plt.title('Training time with NLU and NLG')
plt.xticks(list(CS_data.keys()))

plt.show()
