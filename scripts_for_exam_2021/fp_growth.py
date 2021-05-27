import collections
import itertools
import re

data = []

data = [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], ['a', 'f', 'g'], ['b', 'd',
    'e', 'f', 'j'], ['a', 'b', 'd', 'i', 'k'], ['a', 'b', 'e', 'g'], ['g', 'b']]
# print(data)
data = data
support = 3
# ite frequency
CountItem = collections.defaultdict(int)
for line in data:
    for item in line:
        CountItem[item] += 1

 # dict according to the frequency from large to small, and delete the items with too small frequency
a = sorted(CountItem.items(), key=lambda x: x[1], reverse=True)
for i in range(len(a)):
    if a[i][1] < support:
        a = a[:i]
        break

 # update data, the order of goods for each transaction
for i in range(len(data)):
    data[i] = [char for char in data[i] if CountItem[char] >= support]
    data[i] = sorted(data[i], key=lambda x: CountItem[x], reverse=True)

 # defined good node


class node:
    def __init__(self, val, char):
        self.val = val  # is used to define the current count
        # is used to define what the current character is.
        self.char = char
        self.children = {}  # for storing children
        self.next = None  # for linked lists, linked to another child
        self.father = None  # Search up when building a conditional tree
        # When using the linked list, observe if it has been visited.
        self.visit = 0
        self.nodelink = collections.defaultdict()
        self.nodelink1 = collections.defaultdict()


class FPTree():
    def __init__(self):
        self.root = node(-1, 'root')
        self.FrequentItem = collections.defaultdict(
                int)  # is used to store frequent itemsets
        self.res = []

         # Create a function of the fp tree, data should be in the form of list[list[]], where the internal list contains the name of the item, represented by a string
        def BuildTree(self, data):
            for line in data:  # take the first list, use line to represent
                root = self.root
                for item in line:  # for each item in the list
                    if item not in root.children.keys():  # if item is not in dict
                        root.children[item] = node(
                            1, item)  # Create a new node
                        # is used to search from the bottom up
                        root.children[item].father = root
                    else:
                        # Otherwise, count plus 1
                        root.children[item].val += 1
                        root = root.children[item]  # Go one step down

                        # Create a linked list based on this root
                        if item in self.root.nodelink.keys():  # if this item already exists in nodelink
                    if root.visit == 0:  # If this point has not been visited
                        self.root.nodelink1[item].next = root
                        self.root.nodelink1[item] = self.root.nodelink1[item].next
                        root.visit = 1  # was visited
                    else:  # If this item does not exist in nodelink
                        self.root.nodelink[item] = root
                        self.root.nodelink1[item] = root
                    root.visit = 1
                    print('tree build complete')
        return self.root

    def IsSinglePath(self, root):
                 # print('is it a single path')
        if not root:
            return True
        if not root.children: return True
        a = list(root.children.values())
        if len(a) > 1: return False
        else:
            for value in root.children.values():
                if self.IsSinglePath(value) == False: return False
            return True



         def FP_growth(self,Tree,a,HeadTable): #Tree represents the root node of the tree, a uses a frequent item set represented by the list, and HeadTable is used to represent the header table.
            # We first need to determine whether the tree is a single path, create a single path function IsSinglePath (root)
            if self.IsSinglePath(Tree):#if it is a single path
                # For each combination in the path, denote b, generate mode, b and a, support = b minimum support for nodes in b
            
                         Root, temp = Tree, []#Create an empty list to store
            while root.children:
                for child in root.children.values():
                    temp.append((child.char,child.val))
                    root = child
                         # Generate each combination
            
            ans = []
            for i in range(1,len(temp)+1):
                ans += list(itertools.combinations(temp,i))
            # print('ans = ',ans)
            for item in ans:
                mychar = [char[0] for char in item] + a
                mycount = min([count[1] for count in item])
                if mycount >= support:
                    # print([mychar,mycount])
                    self.res.append([mychar,mycount])
            # print(self.res)
            
        
                 Else:# is not a single path, there are multiple paths
            
            root = Tree
            # print(Tree.char)
                         # Action for each item in the root header table
                         HeadTable.reverse()# first reverses the header table
 
            
                         For (child,count) in HeadTable:#child for characters, count for support
                                 b = [child] + a # new frequent mode
                                 # b's conditional pattern base
                # print(b)
                self.res.append([b,count])
                                 Tmp = Tree.nodelink[child]# At this point, the first node starts from this node, and tmp remains in the linked list.               
                                 Data = []# is used to save the conditional pattern base
                # if b == ['sausage','cream']:
                #    print(root.char)
                                 While tmp:#when tmp always exists
                                         Tmpup = tmp# ready to go up
                    Res = [[],tmpup.val]# is used to save conditional mode
                    
                    while tmpup.father:
                        res[0].append(tmpup.char)
                        tmpup = tmpup.father
                
                                         Res[0] = res[0][::-1]#reverse            
                                         Data.append(res)# conditional pattern base saved
                    tmp = tmp.next
                # if b == ['sausage','cream']: print(2)
                                 # conditional pattern base is completed, stored in data, the next step is to build b's fp-Tree
                
                                 # Statistic word frequency
                CountItem = collections.defaultdict(int)
                for [tmp,count] in data:
                    for i in tmp[:-1]:
                        CountItem[i] += count
                
                for i in range(len(data)):
                                         Data[i][0] = [char for char in data[i][0] if CountItem[char] >= support]#Delete items that do not match
                                         Data[i][0] = sorted(data[i][0],key = lambda x:CountItem[x],reverse=True)#Sort
                    
                # print('2',data)
                                 # Now that the data is ready, all we need to do is construct the condition tree.
                # CountItem1 = collections.defaultdict(int)
                                 Root = node(-1,'root')#Create a root node with a value of -1 and the character root
                                 For [tmp,count] in data:#item is in the form of [list[],count]
         
                                         Tmproot = root# navigate to the root node
                                         For item in tmp:#for every item in tmp
                        # print('123',item)
                        # CountItem1[item] += 1
                                                 If item in tmproot.children.keys():#If the item is already in the tmproot child
                                                         Tmproot.children[item].val += count#update value
                                                 Else:#If this item is not in the tmproot child
                                                         Tmproot.children[item] = node(count,item)#Create a new node
                            Tmproot.children[item].father = tmproot# convenient to find from the bottom up
                                                 Tmproot = tmproot.children[item]#Go one step down
                
                                                 # Create a linked list based on this root
                                                 If item in root.nodelink.keys():#This item exists in nodelink
                            if tmproot.visit == 0:
                                root.nodelink1[item].next = tmproot
                                root.nodelink1[item] = root.nodelink1[item].next
                                tmproot.visit = 1
                                                 Else:#This item does not exist in nodelink
                            root.nodelink[item] = tmproot
                            root.nodelink1[item] = tmproot
                            tmproot.visit = 1
                
 
                                 If root:#if the new condition tree is not empty
                    NewHeadTable = sorted(CountItem.items(),key = lambda x:x[1],reverse=True)
 
                    for i in range(len(NewHeadTable)):
                        if NewHeadTable[i][1] < support:
                            NewHeadTable = NewHeadTable[:i]
                            break
                    
                                         self.FP_growth(root,b,NewHeadTable)#We need to create a new headtable
 
 
                                 # return root#Successfully return the condition tree                
 
    
         Def PrintTree(self,root):#level traversal print tree
        if not root: return
        res = []
        if root.children:
            for (name,child) in root.children.items():
                res += [name+' '+str(child.val),self.PrintTree(child)]
            return res
        else: return 
 
            
                                   
       
obj = FPTree()
root = obj.BuildTree(data)
# print(obj.PrintTree(root))
 
obj.FP_growth(root,[],a) 
print(obj.res)
