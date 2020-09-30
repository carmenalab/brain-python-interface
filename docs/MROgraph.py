#<MROgraph.py>

"""
Draw inheritance hierarchies via Dot (http://www.graphviz.org/)
Author: M. Simionato
E-mail: mis6@pitt.edu
Date: August 2003
License: Python-like
Requires: Python 2.3, dot, standard Unix tools  
"""

import os,itertools,argparse

PSVIEWER='gv'     # you may change these with
PNGVIEWER='evince' # your preferred viewers
PSFONT='Times'    # you may change these too
PNGFONT='Courier' # on my system PNGFONT=Times does not work 

def if_(cond,e1,e2=''):
    "Ternary operator would be" 
    if cond: return e1
    else: return e2
 
def MRO(cls):
    "Returns the MRO of cls as a text"
    out=["MRO of %s:" % cls.__name__]
    for counter,c in enumerate(cls.__mro__):
        name=c.__name__
        bases=','.join([b.__name__ for b in c.__bases__])
        s="  %s - %s(%s)" % (counter,name,bases)
        if type(c) is not type: s+="[%s]" % type(c).__name__
        out.append(s)
    return '\n'.join(out)
      
class MROgraph(object):
    def __init__(self,*classes,**options):
        "Generates the MRO graph of a set of given classes."
        if not classes: raise "Missing class argument!"
        filename=options.get('filename',"MRO_of_%s.ps" % classes[0].__name__)
        self.labels=options.get('labels',2)
        caption=options.get('caption',False)
        self.nometa=options.get('nometa',False)
        setup=options.get('setup','')
        name,dotformat=os.path.splitext(filename)
        format=dotformat[1:] 
        fontopt="fontname="+if_(format=='ps',PSFONT,PNGFONT)
        nodeopt=' node [%s];\n' % fontopt
        edgeopt=' edge [%s];\n' % fontopt
        viewer=if_(format=='ps',PSVIEWER,PNGVIEWER)
        self.textrepr='\n'.join([MRO(cls) for cls in classes])
        caption=if_(caption,
                   'caption [shape=box,label="%s\n",fontsize=9];'
                   % self.textrepr).replace('\n','\\l')
        setupcode=nodeopt+edgeopt+caption+'\n'+setup+'\n'
        codeiter=itertools.chain(*[self.genMROcode(cls) for cls in classes])
        self.dotcode='digraph %s{\n%s%s}' % (
            name,setupcode,'\n'.join(codeiter))
        if os.name == 'nt':
            with open('graph.gv', mode='w') as f:
                f.write(self.dotcode)
            os.system("dot -T%s graph.gv -o %s" %
              (format,filename))
            os.remove('graph.gv')
            os.startfile(filename)
        else:
            os.system("echo '%s' | dot -T%s > %s; %s %s&" %
              (self.dotcode,format,filename,viewer,filename))
    def genMROcode(self,cls):
        "Generates the dot code for the MRO of a given class"
        for mroindex,c in enumerate(cls.__mro__):
            name=c.__name__
            manyparents=len(c.__bases__) > 1
            if c.__bases__:
                yield ''.join([
                    ' edge [style=solid]; %s -> %s %s;\n' % (
                    b.__name__,name,if_(manyparents and self.labels==2,
                                        '[label="%s"]' % (i+1)))
                    for i,b in enumerate(c.__bases__)])
            if manyparents:
                yield " {rank=same; %s}\n" % ''.join([
                    '"%s"; ' % b.__name__ for b in c.__bases__])
            number=if_(self.labels,"%s-" % mroindex)
            label='label="%s"' % (number+name)
            option=if_(issubclass(cls,type), # if cls is a metaclass
                       '[%s]' % label, 
                       '[shape=box,%s]' % label)
            yield(' %s %s;\n' % (name,option))
            if not self.nometa and type(c) is not type: # c has a custom metaclass
                metaname=type(c).__name__
                yield ' edge [style=dashed]; %s -> %s;' % (metaname,name)
    def __repr__(self):
        "Returns the Dot representation of the graph"
        return self.dotcode
    def __str__(self):
        "Returns a text representation of the MRO"
        return self.textrepr

def testHierarchy(**options):
    class M(type): pass # metaclass
    class F(object): pass
    class E(object): pass
    class D(object): pass
    class G(object, metaclass=M): pass
    class C(F,D,G): pass
    class B(E,D): pass
    class A(B,C): pass
    return MROgraph(A,M,**options)

if __name__=="__main__": 
    parser = argparse.ArgumentParser(description='Draw an MRO graph.')
    parser.add_argument('classname', type=str, nargs='+',
                    help='name of the class to draw')
    parser.add_argument('filename', type=str,
                    help='resulting name and filetype')
    parser.add_argument('-l', '--no-labels', dest='labels', 
                    action='store_true', default=False,
                    help='remove numerical labels from each node and edge')
    parser.add_argument('-c', '--caption', dest='caption', 
                    action='store_true', default=False,
                    help='include caption listing each class')
    parser.add_argument('-m', '--hide-meta', dest='meta', 
                    action='store_true', default=False,
                    help='do not include metaclasses in the graph')
    parser.add_argument('-s', '--size', type=str, default="16,12",
                    help='width,height size of graph in inches (default: 16,12)')
    parser.add_argument('-r', '--ratio', type=float, default=0.7)
    parser.add_argument('-e', '--edge-color', type=str, default='blue')
    parser.add_argument('-n', '--node-color', type=str, default='red')

    args = parser.parse_args()
    cls = []
    for c in args.classname:
        pack = c.split('.')
        exec('import ' + '.'.join(pack[0:-1]))
        cls.append(eval(c))
    opt = 'size="%s"; ratio=%f; edge [color=%s]; node [color=%s];' % (args.size, args.ratio, args.edge_color, args.node_color)
    MROgraph(*cls, filename=args.filename, labels=0 if args.labels else 2, caption=args.caption, nometa=args.meta, setup=opt)

#</MROgraph.py>
