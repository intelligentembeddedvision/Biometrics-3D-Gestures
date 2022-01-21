
import sys
from hand_extractor.hand_extractor import *

from math import floor

import pickle


NUMBER_OF_GESTURES = 6
NUMBER_OF_PEOPLE = 1
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
BATCH_SIZE = 32



# These are from hand_extractor
SHOW_IMAGES = True
CLI_VERBOSE = True



def load_data():
    image_path_template = "dataset_in/%.2d_int.bin"
    depth_path_template = "dataset_in/%.2d_dst.bin"

    number_of_people = NUMBER_OF_PEOPLE

    # samples_for_dataset will be a list of tuples (dist_cloud, corresponding_image, number of gesture)
    samples_for_dataset = []

    # for every person
    for person in range(1, number_of_people + 1):
        try:
            image_path = image_path_template %(person)
            depth_path = depth_path_template %(person)

            input_data_format = [NUMBER_OF_GESTURES, -1, IMAGE_HEIGHT, IMAGE_WIDTH]

            image_data = load_bin(image_path, input_data_format)
            depth_data = load_bin(depth_path, input_data_format)

            # for every gesture
            for gesture in range(depth_data.shape[0]):
                # for every frame
                for frame in range(depth_data.shape[1]):
                    depth_sample = depth_data[gesture, frame]
                    image_sample = image_data[gesture, frame]
                    samples_for_dataset.append([depth_sample, image_sample, gesture + 1])
        except:
            pass

    return samples_for_dataset


def save_data(file_name, data_to_save):
    for i, batch in enumerate(data_to_save):
        data = []
        # images = []
        labels = []
        for sample in batch:
            data.append(sample[0])
            # images.append(sample[1])
            labels.append(sample[2])

        data = np.array(data ,'float32')
        # image = np.array(images,'float32')
        labels = np.array(labels ,'uint8')
        save_h5("%s_%.3d.h5 " %(file_name ,i), data, labels)




def split_data(data, split = 0.9):

    samples_per_each_gesture = np.zeros(NUMBER_OF_GESTURES)

    for sample in data:
        samples_per_each_gesture[sample[2] - 1] = samples_per_each_gesture[sample[2] - 1] + 1

    minimum_samples_per_gesture = samples_per_each_gesture.min()

    number_of_batches = floor(minimum_samples_per_gesture * NUMBER_OF_GESTURES / BATCH_SIZE)
    indexes = np.arange(len(data))

    batches = []

    for batch_idx in range(number_of_batches):
        gesture_instances = np.zeros(NUMBER_OF_GESTURES)
        batch_samples = []

        while len(batch_samples) < BATCH_SIZE:
            random_index = floor(np.random.random( ) *len(indexes))
            random_index = indexes[random_index]

            # checking if the gesture has fewer instances than the others (only if not all of them have as many instances)
            if gesture_instances[data[random_index][2] - 1] < gesture_instances.max() or gesture_instances.sum() == gesture_instances.max() * NUMBER_OF_GESTURES:
                batch_samples.append(data[random_index])
                gesture_instances[data[random_index][2] - 1] = gesture_instances[data[random_index][2] - 1] + 1
                np.delete(indexes, random_index)

            if gesture_instances.sum() == BATCH_SIZE:
                batches.append(batch_samples)
                break

    train_size = floor(number_of_batches * split)

    indexes = np.arange(len(batches))
    np.random.shuffle(indexes)

    batches = np.array(batches)
    train_batches = batches[indexes[:train_size]]
    test_batches = batches[indexes[train_size:]]

    return train_batches, test_batches




def on_press(event):
    sys.stdout.flush()
    global keep
    if event.key == 'y':
        keep = True
    elif event.key == 'n':
        keep = False
    else:
        keep = True
    plt.close()



def main():


    samples = load_data()
    processed_samples = []
    failes = []


    for i, sample in enumerate(samples):
        # if i == 30: ##for debug
        #     break
        print("sample: %d - gesture %d " %(i, sample[2]))
        out_hand, out_img, success = run_on_sample(sample[0], sample[1])
        if success:
            processed_samples.append([out_hand, out_img, sample[2]])
        else:
            failes.append([i ,sample[2]])


    # saving the processed samples to load them easier later
    pickle.dump(processed_samples ,open("processed.pickle" ,"wb"))
    pickle.dump(failes ,open("processed_failes.pickle" ,"wb"))

    # checking if the samples are good before saving them
    total_number_of_samples = len(processed_samples)
    total_number_of_samples_filtered = 0

    filtered_samples = []
    discarded_mistakes = []

    to_be_filtered_samples = []
    last_filtered_samples = []
    mistakes = []

    sample_number_per_set = 10
    while True:
        if sample_number_per_set > len(processed_samples):
            sample_number_per_set = len(processed_samples)
        if sample_number_per_set == 0 and len(to_be_filtered_samples) == 0:
            break
        if len(to_be_filtered_samples) == 0:
            to_be_filtered_samples.extend(processed_samples[:sample_number_per_set])
        del processed_samples[:sample_number_per_set]
        total_number_of_samples_filtered = total_number_of_samples - len(processed_samples) - len \
            (to_be_filtered_samples)
        discarded_mistakes.extend(mistakes)
        mistakes = []

        global keep
        keep = True
        for i, sample in enumerate(to_be_filtered_samples):
            imshow(sample[1] ,False ,False)
            plt.gcf().canvas.mpl_connect('key_press_event', on_press)
            plt.title("Sample %d/%d \nPress 'y' to save the image, 'n' to discard it. ('y' is default) " %
            (i + 1 + total_number_of_samples_filtered, total_number_of_samples))
            showPlots()
            if keep:
                last_filtered_samples.append(sample)
            else:
                mistakes.append(sample)

        to_be_filtered_samples = []

        if len(last_filtered_samples) > 0:
            if "y" in input("did you save a bad sample by mistake? 'y' or 'n'\n>>"):
                to_be_filtered_samples.extend(last_filtered_samples)
            else:
                filtered_samples.extend(last_filtered_samples)
                last_filtered_samples = []

        if len(mistakes) > 0:
            if "y" in input("did you discard a good sample by mistake? 'y' or 'n'\n>>"):
                to_be_filtered_samples.extend(mistakes)


    pickle.dump(filtered_samples ,open("filtered.pickle" ,"wb"))
    pickle.dump(discarded_mistakes ,open("filtered_discarded.pickle" ,"wb"))
    train_samples, test_samples = split_data(filtered_samples)


    save_data("dataset_out/train/train_samples" ,train_samples)
    save_data("dataset_out/test/test_samples" ,test_samples)





if __name__ == '__main__':
    main()





