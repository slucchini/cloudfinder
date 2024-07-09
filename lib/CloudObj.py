import numpy as np
from collections.abc import Iterable

### TODO:
# update < and > to take iterables?

class CloudObj():
    def __init__(self,members=[],snapnum=0):
        # list(np.unique(members)).sort()
        self.members = np.unique(members)
        self.snapnum = snapnum
    
    def get_snapmask(self,sids,argsort):
        mask = np.zeros(len(sids),dtype=bool)
        idcheck = [m in sids for m in self.members]
        idmask = np.searchsorted(sids,self.members[idcheck])
        pos_to_change = np.arange(0,len(mask))[argsort][idmask]
        mask[pos_to_change] = True
        return pos_to_change
    
    def __repr__(self):
        if ((len(self.members)>0) and hasattr(self,'snapnum')):
            return "CloudObj ({}) {} <len {}>".format(self.snapnum,self.members[0],len(self.members))
        elif (len(self.members)>0):
            return "CloudObj {} <len {}>".format(self.members[0],len(self.members))
        else:
            return "CloudObj -- <len 0>"
    
    def __len__(self):
        return len(self.members)
    
    def __eq__(self,other):
        if isinstance(other, Iterable):
            return np.array([self.__hash__() == o.__hash__() for o in other])
        else:
            return self.__hash__() == other.__hash__()
    
    def __lt__(self,other):
        # isiter = True
        # if not isinstance(other, Iterable):
        #     isiter = False
        #     other = [other]
        test1 = self.snapnum < other.snapnum
        members_test = (len(self) > 0) & (len(other) > 0)
        test2 = (self.snapnum == other.snapnum) & members_test & (self.members[0] < other.members[0])
        test3 = (self.snapnum == other.snapnum) & members_test & (self.members[0] == other.members[0]) & (len(self) < len(other))
        return test1 | test2 | test3

    def __gt__(self,other):
        test1 = self.snapnum > other.snapnum
        members_test = (len(self) > 0) & (len(other) > 0)
        test2 = (self.snapnum == other.snapnum) & members_test & (self.members[0] > other.members[0])
        test3 = (self.snapnum == other.snapnum) & members_test & (self.members[0] == other.members[0]) & (len(self) > len(other))
        return test1 | test2 | test3
    
    def __le__(self,other):
        return (self < other) | (self == other)
    
    def __ge__(self,other):
        return (self > other) | (self == other)

    def __hash__(self):
        # self.members.sort()
        try:
            return hash(frozenset([self.snapnum]+self.members))
        except:
            return hash(frozenset(self.members))