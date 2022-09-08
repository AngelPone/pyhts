Define Hierarchy
================


Cross-sectional Hierarchy
-------------------------

There are three kinds of cross-sectional hierarchy: regular, grouped or mix of both (Hyndman & Athanasopoulos, 2021).
The last type is most of the case in practice. In this tutorial, we will destruct the last type of hierarchy, and show how
complex hierarchical structures can be constructed using :class:`~pyhts.Hierarchy.new()`.

Almost every hierarchy can be seen as a combination of multiple regular hierarchies. Here, regular hierarchy refers to
hierarchy that can be represented by a tree, where nodes in each level are completely unique. For example, for the product
category case, it can be "Category" -> "Subcategory" -> "item". The geographical hierarchy is also a classical example.
Another example is the file system.

Mixing two or more regular hierarchies construct a complex but widely used hierarchy. Every "level" in this hierarchy is 
interaction of two or more levels in different regular hierarchies. Here is an example, still the product case.

For "category", the regular hierarchy can be `total`, `category`, `subcategory` and `item`.
Each product can also be sold at multiple locations. 
For "location", the regular hierarchy can be `total`, `state`, `region`, `store`.

Mixing these two hierarchies gives following levels, totally 4 * 4 = 16 levels:

* total = total (category) x total (location)
* category = category x total (location)
* subcategory = subcategory x total (location)
* item = item x total (location)
* state = total (category) x state
* region = total (category) x region
* store = total (category) x store
* category x state
* category x region
* category x store
* subcategory x state
* subcategory x region
* subcategory x store
* item x state
* item x region
* item x store (bottom level)

Another example of this kind of hierarchy is the file system with tags, while usually tags only have one level except the total level.

To construct the product hierarchy using :class:`pyhts.Hierarchy.new()`, use the following statement:

.. code-block:: Python

    Hierarchy.new(df, structures=[('category', 'subcategory', 'item'), ('state', 'region', 'store')])


You can also specify :code:`excludes` and :code:`includes` to exclude some levels or only include some levels use the same rule.


.. autoclass:: pyhts.Hierarchy
    :members: new



Reference
---------

[1] Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). Otext. https://otexts.com/fpp3/
