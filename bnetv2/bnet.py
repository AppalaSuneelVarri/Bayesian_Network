import pandas as pd
import sys
import itertools

def parse_args(args):
    query = {}
    evidence = {}
    i = 2
    while i < len(args):
        if args[i] == "given":
            i += 1
            break
        query[args[i][0]] = int(args[i][1] == 't')  
        i += 1

    while i < len(args):
        evidence[args[i][0]] = int(args[i][1] == 't') 
        i += 1

    return query, evidence

class BayesianNetwork:
    def __init__(self, structure):
        self.structure = structure
        self.data = None
        self.prob_tables = {}

    def initialize_prior_probabilities(self, child, parents):
        prior_prob_table = {}
        all_combinations = list(itertools.product([0, 1], repeat=len(parents)))
        for combination in all_combinations:
            prior_prob_table[tuple(combination)] = {0: 0.5, 1: 0.5}
        return prior_prob_table

    def fit(self, data):
        self.data = data
        for child, parents in self.structure.items():
            parents_data = data[parents]
            unique_parent_combinations = parents_data.drop_duplicates().values

            child_prob_table = self.initialize_prior_probabilities(child, parents)
            for combination in unique_parent_combinations:
                parents_mask = (parents_data == combination).all(axis=1)
                child_given_parents = data[child][parents_mask]
                counts = child_given_parents.value_counts(normalize=True)
                child_prob_table[tuple(combination)].update(counts.to_dict())

            self.prob_tables[child] = child_prob_table
        
    def joint_probability(self, evidence):
        joint_prob = 1.0
        for var in evidence.keys():
            if var in self.structure:
                child_value = evidence[var]
                parents = self.structure[var]
                parents_values = tuple(evidence.get(parent) for parent in parents)
                if None in parents_values:
                    joint_prob_child_given_parents = 0
                    parents_with_none = [parent for parent in parents if evidence.get(parent) is None]
                    for parent_combination in itertools.product(*[data[parent].unique() for parent in parents_with_none]):
                        updated_parents_values = tuple(parent_combination[parents_with_none.index(parent)] if value is None else value for parent, value in zip(parents, parents_values))
                        joint_prob_child_given_parents += self.prob_tables[var][updated_parents_values][child_value]

                    joint_prob *= joint_prob_child_given_parents
                else:
                    joint_prob *= self.prob_tables[var][parents_values][child_value]
            else:
                child_value = evidence[var]
                counts = self.data[var].value_counts(normalize=True)
                joint_prob *= counts[child_value]
        return joint_prob
    
    def print_table(self, cpt_dict):
        for var in sorted(cpt_dict.keys()):
            var_states = sorted(cpt_dict[var].keys())
            headers = ['{}({})'.format(var, state) for state in var_states]
            print('+{}+'.format('+'.join(['-'*(len(header)+2) for header in headers])))
            print('|{}|'.format('|'.join([' {} '.format(header) for header in headers])))
            print('+{}+'.format('+'.join(['-'*(len(header)+2) for header in headers])))
            for row in sorted(cpt_dict[var][var_states[0]].keys()):
                values = []
                for state in var_states:
                    values.append('{:.5f}'.format(cpt_dict[var][state][row]))
                print('|{}|'.format('|'.join([' {} '.format(value) for value in values])))
                print('+{}+'.format('+'.join(['-'*(len(header)+2) for header in headers])))
            print('\n')
    

    def show_cpt(self):
        cpt = self.prob_tables
        sumb = sum(self.data["B"])
        sumc = sum(self.data["C"])
        cpt["B"]={}
        cpt["C"]={}
        cpt["B"][()] = {0: (1 - (sumb)/len(self.data)), 1: (sumb)/len(self.data)}
        cpt["C"][()] = {0: (1 - (sumc)/len(self.data)), 1: (sumc)/len(self.data)}
        return cpt

class VariableElimination:
    def __init__(self, model):
        self.model = model

    def query(self, variables, evidence):
        options = [0, 1]
        joint_probability = 0
        marginal_probability = 0

        for b in options:
            for g in options:
                for c in options:
                    for f in options:
                        
                        full_evidence = {'B': b, 'G': g, 'C': c, 'F': f}
                        
                        jpd_value = self.model.joint_probability(full_evidence)

                        if all(evidence.get(var, var_value) == var_value for var, var_value in full_evidence.items()):
                            marginal_probability += jpd_value

                            
                            if all(variables.get(var, var_value) == var_value for var, var_value in full_evidence.items()):
                                joint_probability += jpd_value

        conditional_probability = joint_probability / marginal_probability

        return conditional_probability


    def _generate_combinations(self, variables):
        if not variables:
            yield {}
        else:
            var = variables.pop()
            for val in data[var].unique():
                for combination in self._generate_combinations(variables.copy()):
                    yield {**combination, var: val}

if __name__ == "__main__":


    if len(sys.argv) < 2:
        print("\nUsage: python bnet.py <training_data>")
        print("\nUsage: python bnet.py <training_data> <Bt/Bf> <Gt/Gf> <Ct/Cf> <Ft/Ff>")
        print("\nUsage: python bnet.py <training_data> <query variable values> [given <evidence variable values>]")
        sys.exit(1)

    data = pd.read_csv(sys.argv[1], header=None, sep='\s+')
    data.columns = ['B', 'G', 'C', 'F']
    model = BayesianNetwork(structure={'G': ['B'], 'F': ['G', 'C']})
    model.fit(data)

    if (len(sys.argv) == 2):
        model.print_table(model.show_cpt())
        sys.exit(1)

    query, evidence = parse_args(sys.argv)
    # print("Query: ", query)
    # print("Evidence: ", evidence)
    if any(key in evidence.keys() for key in query.keys()) or any(key in query.keys() for key in evidence.keys()):
        print("Error: query and evidence variables must not be the same variables")
        sys.exit(1)

    infer = VariableElimination(model=model)
    result_prob = infer.query(variables=query, evidence=evidence)
    
    print(f"P({query} | {evidence}) = {result_prob}")

