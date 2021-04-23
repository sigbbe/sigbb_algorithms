# FP-Growth Algorithm - Discovering frequent itemsets without candidate generation

## Definitions

**Frequent itemset**: An itemset *X* that occurs at least as frequently as a predetermined minimum support count, *min_sup*.

**Closed frequent itemset**: An itemset *X* is **closed** in a data set *D* if there exists no proper super-itemset *Y* such that *Y* has the same support count as *X* in.  

**Maximal frequent itemset**: An itemset *X* is a **maximal frequent itemset** (or **max-itemset**) in a dataset *D* if *X* is frequent, and there exists no super-itemset *Y* such that *X ⊂ Y* and *Y* is frequent in *D*.

## Pseudo code

Procedure FP growth(Tree, α)  
    (1) **if** Tree contains a single path P then  
    (2)&nbsp;&nbsp;&nbsp;&nbsp; **for each** combination (denoted as β) of the nodes in the path P  
    (3)&nbsp;&ensp;&ensp;&ensp;&nbsp;&ensp;&ensp;&ensp;         generate pattern β ∪α with support count = minimum support count of nodes in β;  
    (4)&nbsp;&ensp;&ensp;&ensp; **else for each** *a_i* in the header of *Tree* {  
    (5)&nbsp;&ensp;&nbsp;&ensp;&ensp;&ensp;&nbsp;&ensp;     generate pattern *β* = *a_i* ∪ α with *support count* = *a_i*.support count;  
    (6)&nbsp;&ensp;&nbsp;&ensp;&ensp;&ensp;&nbsp;&ensp; construct β’s conditional pattern base and then *β*’s conditional FP tree *Treeβ*;  
    (7)&nbsp;&ensp; **if** *Treeβ* != ∅ **then**  
    (8)&nbsp;&ensp; call **FP growth**(*Treeβ* , β);  
    &nbsp;&ensp;&nbsp;&ensp;&nbsp;&ensp;&nbsp;}

{A}
{FA}
{CA}
{FCA}
{M}
{MA}
{FM}
{FMA}
{CM}
{FCM}
{CMA}
{FCMA}
{P}
{CP}
{B}
{C}
{F}
{CF}

Time in terms of AM and PM:
Brightness as measured by a light mete:
Brightness as measured by people’s judgments:
Angles as measured in degrees between 0 and 360:
Bronze, Silver, and Gold medals as awarded at the Olympics:
Height above sea level:
Number of patients in a hospital:
ISBN numbers for books. (Look up the format on the Web:
Ability to pass light in terms of the following values: opaque, translucent, transparent:
Military rank:
Distance from the center of campus:
Density of a substance in grams per cubic centimeter:
Coat check number. (When you attend an event, you can often give your coat to someone who, in turn, gives you a number that you can use to claim your coat when you leave.):
