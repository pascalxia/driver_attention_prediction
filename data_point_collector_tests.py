import json
from data_point_collector import *

DATA_DIR = "./camera_images"

def run_tests():
    # Sample rates to test:
    sample_rates = [3,  # The default prediction rate. Should only have 1 group.
                    7,  # Not a multiple of 3 -- should have 7 groups
                    6,  # Multiple of 3 -- should have 6/3=2 groups
                    2]  # Less than 3 fps

    for rate in sample_rates:
        print("\n\nTESTING SAMPLE RATE: " + str(rate) + "\n\n")
        test_group_assignment(rate)

# Tests that frames get assigned to the correct group based on sample rate.
def test_group_assignment(sample_rate):
    PREDICTION_RATE = 3
    NUM_SECONDS = 5

    effective_rate = sample_rate if sample_rate % 3 == 0 \
                        else sample_rate * PREDICTION_RATE


    clear_old_files(DATA_DIR)
    write_dummy_files(effective_rate * NUM_SECONDS, ".jpg", \
                        DATA_DIR, "image")
    result = get_data_point_names(".", in_sequences=True, sampleRate=sample_rate)

    keys = list(range(len(result)))
    dict_result = dict(zip(keys, result))
    print("GROUPS", json.dumps(dict_result, indent=2))
    print("Return value", result)

def leftpad(text, length, pad_char=0):
    while len(text) < length:
        text = str(pad_char) + text
    return text

# Writes dummy .jpg files to test the data point grouping.
def write_dummy_files(num_files, extension, directory, name_prefix="test"):
    for i in range(num_files):
        f = open(directory + "/" + name_prefix + leftpad(str(i), 5) + extension,"w+")
        f.close()

def clear_old_files(directory):
    files = os.listdir(directory)
    for item in files:
        if item.endswith(".jpg"):
            os.remove(os.path.join(directory, item))
