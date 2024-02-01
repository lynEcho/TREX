class GroupInfo:
    def __init__(self, group_freqs, major, minor, category=None, tgt_p=None):
        """
        Args:
            group_freqs(pandas.Series):
                Group names with their frequencies in the population.
            category(str):
                The name of what the group represents.  For example, the gender of book authors.
            major(str):
                The name of the majority group.
            minor(str):
                The name of the minority group.
            tgt_p(float or None):
                The target probability for binomial fairness.
        """
        self.group_freqs = group_freqs
        self.category = category
        self.major = major
        self.minor = minor

        if tgt_p is None:
            # estimate p from group_freqs
            # TODO review this decision
            self.tgt_p_binomial = group_freqs.loc[minor] / (group_freqs[minor] + group_freqs[major])
            
        else:
            self.tgt_p_binomial = tgt_p
