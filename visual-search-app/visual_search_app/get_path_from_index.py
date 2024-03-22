import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def get_image_path(index):
    with open('C:\\Users\\kshubhan\\Downloads\\Proxzar-CV-work-updated\\Proxzar-CV-work-main\\Proxzar\\visual_search_app\\metadata-files\\vgg19\\image_data_features.pkl', 'rb') as file:
        data = pickle.load(file)
    return data['images_paths'][index]

def plot_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    index = input("Enter the index: ")
    image_path = get_image_path(int(index))
    plot = input("Do you want to plot the image? (yes/no): ")
    if plot.lower() == "yes":
        plot_image(image_path)
        print(f"Image path: {image_path}")
    else:
        print(f"Image path: {image_path}")