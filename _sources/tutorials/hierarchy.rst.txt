Define Hierarchy
================


- :class:`pyhts.hierarchy.Hierarchy.from_node_list()` constructs hierarchy from a list of lists that contains number of children nodes of all non-leaf nodes from root to leaves of the tree.

- :class:`pyhts.hierarchy.Hierarchy.from_chars()` constructs hierarchy from names of bottom series which use letter that has specific length to represent their parent nodes.

- :class:`pyhts.hierarchy.Hierarchy.from_long()` constructs hierarchy from long data table.

- :class:`pyhts.hierarchy.Hierarchy.from_balance_group()` construct balance group time series that have multiple group types and all groups of each group type contains