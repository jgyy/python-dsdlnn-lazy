"""
show images script
"""
from numpy.random import choice
from matplotlib.pyplot import imshow, title, show
from util import get_data

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
    """
    main function
    """
    x_datas, y_datas, _, _ = get_data(balance_ones=False)
    while True:
        for i in range(7):
            x_data, y_data = x_datas[y_datas==i], y_datas[y_datas==i]
            n_data = len(y_data)
            j = choice(n_data)
            imshow(x_data[j].reshape(48, 48), cmap='gray')
            title(label_map[y_data[j]])
            show()
        prompt = input('Quit? Enter Y:\n')
        if prompt.lower().startswith('y'):
            break


if __name__ == '__main__':
    main()
