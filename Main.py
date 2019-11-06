import csv
import logging
import sys
from ast import literal_eval
from datetime import datetime, timedelta
from statistics import mean

import numpy as np
import pandas as pd
import yaml

# create log file
logging.basicConfig(filename=f"Logs/{datetime.now().strftime('%d-%m-%y %H-%M')} log.log", level=logging.INFO)
# disable tensor flow future warning logging
logging.getLogger('tensorflow').disabled = True

# disable tensor flow future warning logging
sys.stderr = open('errorlog.log', 'a+')
from luminoth import Detector, read_image


class MainTest:
    def __init__(self):
        """
    The main test class, given a yaml file containing a checkpoint name, an image database name
     and a list of thresholds, runs a set of basic tests in order verify the perception results.
     The output is both a log file and an optional result csv contained in their respective folders.
        """
        logging.info(f"\n\n{str(datetime.now())} New Session")
        with open('Input.yml', 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        # check if input checkpoint name is of the listed in 'lumi checkpoint list'
        self.checkpoint_list = ['accurate', 'fast']
        if cfg['Input']['checkpoint'] in self.checkpoint_list:
            self.cname = cfg['Input']['checkpoint']
        else:
            logging.error('Input checkpoint name does not exist in database, setting default parameters.')
            self.cname = 'fast'
        # initiate lists of images and labels to be added
        self.test_images = []
        self.test_labels = []
        self.image_names = []
        # initiate the pass/fail criteria
        self.pass_ = True
        # initiate thresholds dictionary
        self.thresholds = {}
        try:
            for key in cfg['Thresholds'].keys():
                self.thresholds[key] = cfg['Thresholds'][key]
            self.thresholds['runtime'] = timedelta(seconds=self.thresholds['runtime'])
        except TypeError:
            logging.error('Wrong threshold list input, setting default parameters.')
            self.thresholds = {'recall': 0.9, 'probability': 0.9, 'precision': 0.9,
                               'runtime': timedelta(seconds=1)}
        self.im_database = cfg['Input']['image database']
        self.results = pd.DataFrame(
            columns=('Recall', 'Probability', 'Precision', 'Runtime', 'Image', 'Checkpoint'))

    def get_images(self):
        """
    Reads images and label from a tsv database: row 1 contains the images, row 2 contains dictionaries with labels.
    Images are stored in the 'images' folder. In the future this will be made with an online pre-made database.
        """
        with open(self.im_database) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                self.test_images.append(read_image(f"images/{row[0]}"))
                self.test_labels.append(literal_eval(row[1]))
                self.image_names.append(row[0])

    def im_test(self, detector: Detector, image: np.ndarray, labels: dict) -> tuple:
        """
    tests if the image was labeled correctly with an recall and probability higher than the threshold.

        :return: mean recall, probability, precision and runtime.
        :param detector: a luminoth detector object with the chosen checkpoint.
        :param image: a numpy.ndarray containing the test image.
        :param labels: a dictionary containing the labels of the objects in the image.
        """
        start = datetime.now()
        objects = detector.predict(image)
        # initiate a dictionary for found labels, and lists for accuracies and probabilities.
        time = datetime.now() - start
        logging.info(
            f"Run time is {time.total_seconds()}, the threshold is {self.thresholds['runtime'].total_seconds()}")
        test_dict = {}
        tp = 0
        fp = 0
        fn = 0
        probabilities = []
        # fill test dictionary with labels found.
        for dic in objects:
            if dic['label'] in test_dict:
                test_dict[dic['label']] += 1
            else:
                test_dict[dic['label']] = 1
            probabilities.append(dic['prob'])
        # check false negative.
        for key in labels.keys():
            if key in test_dict:
                if test_dict[key] - labels[key] == 0:
                    tp += labels[key]
                elif labels[key] > test_dict[key]:
                    tp += test_dict[key]
                    fn += labels[key] - test_dict[key]
                else:
                    tp += labels[key]
                    fp += test_dict[key] - labels[key]
                logging.info(f"Found {test_dict[key]} {key}s out of {labels[key]}.")
            else:
                fn += labels[key]
                logging.warning(f"Found {0} {key}s out of {labels[key]}.")
        # check false positive
        for key in test_dict.keys():
            if key not in labels:
                fp += test_dict[key]
                logging.warning(f"False positive label was detected.")
        # calculate recall and probability.
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        mean_prob = mean(probabilities)
        logging.info(f"Recall is {round(recall, 2)}, the threshold is {self.thresholds['recall']}.")
        logging.info(f"Precision is {round(precision, 2)}, the threshold is {self.thresholds['precision']}.")
        logging.info(f"Mean probability is {round(mean_prob, 2)}, the threshold is {self.thresholds['probability']}.")
        if recall >= self.thresholds['recall'] and mean_prob >= self.thresholds['probability'] and time <= \
                self.thresholds['runtime']:
            logging.warning("The test has passed successfully.")
        else:
            logging.warning("The test has failed.")

        return recall, mean_prob, precision, time

    def run(self):
        """
    Run the image test on all images in the database.
        """
        logging.info("Executing main test.")
        # initialize lists of accuracies, probabilities and runtimes
        recall = []
        probabilities = []
        times = []
        precision = []
        detector = Detector(checkpoint=self.cname)
        # run the image test for every image in the database
        for i in range(0, len(self.test_images)):
            logging.info(f"Executing image test number {i + 1}:")
            rec, prob, fp, time = self.im_test(detector, self.test_images[i], self.test_labels[i])
            precision.append(fp)
            recall.append(rec)
            probabilities.append(prob)
            times.append(time)
            # self.results.append(self.im_test(detector, self.test_images[i], self.test_labels[i]))

        # create a DataFrame of the data
        df = pd.DataFrame(
            {'Recall': recall, 'Probability': probabilities, 'Precision': precision, 'Runtime': times,
             'Image': self.image_names, 'Checkpoint': [self.cname] * len(recall)})
        self.results = self.results.append(df)
        self.log_results()

    def log_results(self):
        """
    Log the success or failure of the main test calculated using the average of all tests.
        """
        means = self.results.mean()
        if means[0] >= self.thresholds['recall']:
            logging.info(f"The total mean recall is {round(means[0], 2)} and is higher"
                         f" than the {self.thresholds['recall']} threshold.")
        else:
            logging.info(f"The total mean recall is {round(means[0], 2)} and is lower"
                         f" than the {self.thresholds['recall']} threshold.")
            self.pass_ = False
        if means[1] >= self.thresholds['probability']:
            logging.info(f"The total mean probability is {round(means[1], 2)} and is higher"
                         f" than the {self.thresholds['probability']} threshold.")
        else:
            logging.info(f"The total mean probability is {round(means[1], 2)} and is lower"
                         f" than the {self.thresholds['probability']} threshold.")
            self.pass_ = False
        if means[2] > self.thresholds['precision']:
            self.pass_ = False
            logging.info(f"The precision is {round(means[2], 2)} and is higher"
                         f" than the {self.thresholds['precision']} threshold.")
        else:
            logging.info(f"The precision is {round(means[2], 2)} and is lower"
                         f" than the {self.thresholds['precision']} threshold.")
        if means[3] <= self.thresholds['runtime']:
            logging.info(f"The total mean runtime is {means[3].total_seconds()} and is lower"
                         f" than the {self.thresholds['runtime'].total_seconds()} threshold.")
        else:
            logging.info(f"The total mean runtime is {means[3].total_seconds()} and is higher"
                         f" than the {self.thresholds['runtime'].total_seconds()} threshold.")
            self.pass_ = False

        if self.pass_:
            logging.info("Test has passed all parameters.")
        else:
            logging.warning("Test has failed.")

    def export_result(self):
        self.results.to_csv(f"Results/{datetime.now().strftime('%d-%m-%y %H-%M')} results.csv", header=True,
                            index=False)


if __name__ == '__main__':
    mytest = MainTest()
    mytest.get_images()
    mytest.run()
    mytest.export_result()
