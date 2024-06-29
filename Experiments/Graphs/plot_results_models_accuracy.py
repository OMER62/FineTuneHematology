import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    labels = ['base', 'rag', 'pipeline']
    num_bars = 3
    # the data is going to be accuracy base, accuracy rag, accuracy pipeline, accuracy ranker
    colors = ["darkslategrey", "dimgrey", "beige", "lightgrey"]

    data = {
        'gpt-3.5':[0.48, 0.48, 0.48],
        'llama2-7b': [0.26, 0.2, 0.22],
        'zephyr': [0.3, 0.32, 0.34],
        'mistral': [0.32, 0.34, 0.37]
        # openai embedding
        #'mistral': [0.32, 0.45, 0.47]
        # medical textbook
        #'mistral':[0.32, 0.32, 0.34]
    }

    # Calculate bar width and positions
    x = np.arange(len(labels))  # Label locations
    width = 0.2  # Width of the bars

    # Create subplots
    fig, ax = plt.subplots()

    # Plot bars for each group
    for i, (label, values) in enumerate(data.items()):
        ax.bar(x + i * width - width, values, width, label=f'{label}', color=colors[i])

    # Add labels and title
    ax.set_xlabel('inference')
    ax.set_ylabel('accuracy')
    ax.set_title('Hematology-index-QATexts')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()

    # Show plot
    plt.show()
