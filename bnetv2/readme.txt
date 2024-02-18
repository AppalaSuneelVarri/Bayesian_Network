Name: APPALA SUNEEL VARRI  
UTA ID: 1002111813

Programming Language:
Python 3.8

Code Structure:
The code contains the following classes and functions:

1. parse_args(args): Parses command line arguments and returns query and evidence dictionaries.

2. BayesianNetwork: Class representing a Bayesian Network, containing structure, data, and probability tables.
   - `__init__(self, structure)`: Initializes the Bayesian Network with the given structure.
      here structure is the relationship between the nodes.
   - `fit(self, data)`: Fits the Bayesian Network with the given data.
   - `joint_probability(self, evidence)`: Calculates the joint probability for given evidence.
   - `show_cpt(self)`: gives cpt table data, in dictionary format to the print_table function.
   - `print_table(cpt_dict)`: prints the CPT data, with the provided cpt_dict table.

3. VariableElimination: Class representing the variable elimination algorithm for Bayesian Networks.
   - `__init__(self, model)`: Initializes the VariableElimination instance with the given Bayesian Network model.
   - `query(self, variables, evidence)`: Performs the query on the Bayesian Network.
   - `_generate_combinations(self, variables)`: Generates all possible combinations for the given variables.

How to run the code:

1. Install pandas library if not already installed:

pip install pandas

2. Run the code as follows depending on the task:

Task 1: Learn the conditional probability tables from the training data:

python bnet.py <training_data.txt>

Sample invocation :  python bnet.py training_data.txt

Output: 
Provides output probability values in the form of tabels where 
B() table has the values of 1st row with no baseball game on TV (B is false), 
and second row is if there is a baseball game on TV

Task 2: Calculate the joint probability distribution for given variables:

python bnet.py <training_data.txt> <Bt/Bf> <Gt/Gf> <Ct/Cf> <Ft/Ff>
Sample invocation : python bnet.py training_data.txt Bf Gt Ct Ff

Gives the calculated probability.

Task 3: Calculate the probability for any event given evidence (if given):

python bnet.py <training_data.txt> <query variable values> [given <evidence variable values>]

Sample invocation : python bnet.py training.txt Bf Ct given Ff
Output : Gives the calculated probability values.

Replace `<training_data>` with the path to the training data file, and `<query variable values>` and `<evidence variable values>` with the appropriate values as per the description.

Running on ACS Omega:
The code should run on ACS Omega without any issues.

