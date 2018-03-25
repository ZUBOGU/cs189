# Homework 1
There are total 5 code files:
digit_training.py
find_parameter_C_for_digit.py
find_parameter_C_for_spam.py
kaggle_digit_submission.py
kaggle_spam_submission.py

Put the data folder inside this folder

    $ cd code_directory

Run

    $ python digit_training.py

It will ouput error rate in terminal and show the graph for problem 1 in a separate window.
By closing the separate graph window, it will show the confusion matrix for 100, 200, 500, 1000,
2000, 5000, 10000 smaples continuously for problem 2.
In the end, the terminal will report the validation error rate for an SVM with the optimal C for problem 3.

Run 

    $ python find_parameter_C_for_digit.py

To experiment and find the optimal value C, by change the C value in the line.

	clf = svm.LinearSVC(C = xxx)


The one have the highest accuray reported in the terminal is the optimal value for C.

Do the same experiment for problem 4 by run

    $ python find_parameter_C_for_spam.py

The accuracy using cross validation will report on termnial for problem 4. 

Run

    $ python kaggle_digit_submission.py
or
    $ python kaggle_digit_submission.py

It will generate CSV formate file for Kaggle submisiions.