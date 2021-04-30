# Assignment 5 - Decision Trees

## Description

A small computer retailer, wants to predict/decide if a customer should get a PC on
credit, or not. Table 2 contains examples of the decisions the company has made in the past. Assume
that each customer record has five attributes as follows: CustomerID, Age, Income, Student,Creditworthiness, PC_on_Credit.

Classes:

We use the decision tree algorithm along with the gini index to decide the attribute which should be the splitting attribute.

In our example  we have 2 classes (PC_on_Credit: yes/no) and 20 objects. 12 objects are in class 1 (yes), and 8 objects in class 2 (no). We denote this distribution as (12, 8).

The Gini index would be: 1 - [(12/20)^2 + (8/20)^2] = 0.6247 i.e. costbefore = Gini(19,21,40) = 0.6247
