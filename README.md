# Bayesian_Network

Problem Statement: Bayesian Network Inference

Develop a Python program that performs inference on a Bayesian network given a set of variables and evidence. The program should support the following functionalities:

Training Data Processing:
Read training data from a CSV file specified as a command-line argument.
The CSV file contains binary values (0 or 1) representing variables B, G, C, and F.
Bayesian Network Model:
Define a Bayesian network model with the following structure:
Variable G is dependent on variable B.
Variable F is dependent on variables G and C.
Implement methods to fit the model using the training data and calculate prior probabilities.
Inference:
Perform inference on the Bayesian network to calculate conditional probabilities.
Support two types of queries:
Joint probability query: Calculate the joint probability of specific variable values given no evidence.
Conditional probability query: Calculate the conditional probability of specific variable values given evidence.
Unit Testing:
Develop unit tests to validate the functionality of the Bayesian network model and inference methods.
Test cases should cover the calculation of conditional probabilities and joint probabilities.
Command-line Interface:
Accept command-line arguments to specify the training data file path and query variables with optional evidence.
Display the calculated probabilities based on the provided query and evidence.
Error Handling:
Handle errors gracefully, such as incorrect command-line arguments and conflicting query and evidence variables.
Output Formatting:
Format the output to display probabilities with a precision of four decimal places.
Ensure that the program adheres to best practices in terms of code readability, documentation, and modularity. Additionally, implement error handling to provide informative messages in case of unexpected input or errors during execution.

The provided code serves as a foundation for implementing the Bayesian network inference program. You are required to extend and refine the code to meet the specified requirements. Additionally, ensure that the unit tests cover all relevant scenarios to validate the correctness of the implementation.
