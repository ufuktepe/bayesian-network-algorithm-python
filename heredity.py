import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}

# Maps number of genes to the probability of passing the gene to child
PASSING_PROBS = {

    # If parent doesn't have the gene, it can only be passed by mutation
    0: PROBS["mutation"],

    # If parent has one gene, it'll be passed with probability 0.5
    1: 0.5,

    # If parent has two genes, it'll be passed unless there is a mutation
    2: 1 - PROBS["mutation"]
}


class Person:
    """
    Represents a person that contains information about the number of
    copies of the gene that the person has, the probabilities that the
    person's parents pass the gene and whether the person has the trait or not.
    """

    def __init__(self, name, mother, father, one_gene, two_genes, have_trait):
        # Number of copies of the gene that the person has
        self.gene_count = count(name, one_gene, two_genes)

        # Probability that the person's mother passes the gene
        self.p_m = PASSING_PROBS.get(count(mother, one_gene, two_genes), None)

        # Probability that the person's father passes the gene
        self.p_f = PASSING_PROBS.get(count(father, one_gene, two_genes), None)

        # Boolean attribute indicating whether the person has the trait or not
        self.has_trait = name in have_trait

    def calculate_probability(self):
        """
        Calculates the probability based on the number of copies of the gene
        the person has and whether the person exhibits the trait or not.
        """
        if self.p_m is None:
            # Get the unconditional probability if the person has no parents
            gene_prob = PROBS['gene'][self.gene_count]
        elif self.gene_count == 0:
            # Neither the mother nor the father passes the gene
            gene_prob = (1 - self.p_m) * (1 - self.p_f)
        elif self.gene_count == 1:
            # Only the mother or only the father passes the gene
            gene_prob = self.p_m * (1 - self.p_f) + self.p_f * (1 - self.p_m)
        else:
            # Both parents pass the gene
            gene_prob = self.p_m * self.p_f

        return gene_prob * PROBS['trait'][self.gene_count][self.has_trait]


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob = 1

    for person_data in people.values():
        person = Person(person_data['name'], person_data['mother'],
                        person_data['father'], one_gene, two_genes, have_trait)
        # Multiply the probability for the person with joint probability
        joint_prob *= person.calculate_probability()

    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person, person_probs in probabilities.items():
        gene_count = count(person, one_gene, two_genes)
        person_probs['gene'][gene_count] += p
        person_probs['trait'][person in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person_probs in probabilities.values():
        for prob_distribution in person_probs.values():
            total = sum(prob_distribution.values())
            # Normalize the probabilities
            for k, v in prob_distribution.items():
                prob_distribution[k] = v / total


def count(person, one_gene, two_genes):
    """
    Returns the number of genes that the given person has.
    """
    if person is None:
        return None
    return 2 if person in two_genes else 1 if person in one_gene else 0


if __name__ == "__main__":
    main()
