import  Network as nt
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog


def get_links_with_coord(xmlData):

    source_target = []
    number_node = 0  #start from 1
    # get the id
    nodeIDList = []
    # get the coordinates
    xyList = []
    # nameID list
    nameIDList = []

    #itterate thru the link
    for link in xmlData[0][1]:
        #get the link source and target name
        linkSource = link.find('source').text
        linkTarget = link.find('target').text

        ## get [(source,target),]
        source_target.append((linkSource,linkTarget))

    for node in xmlData[0][0]:

        #get the node name
        nodeName = node.attrib.get('id')
        #append it to nodeIDList
        nodeIDList.append(nodeName)
        #for id referring
        nameIDList.append((nodeName,number_node))

        #count node
        number_node = number_node + 1

        # getting the coordinate
        coordObj = node.find('./coordinates')
        xCoord = coordObj.find('x').text
        yCoord = coordObj.find('y' ).text
        xyTuple = (xCoord,yCoord)
        xyList.append(xyTuple)

    value = {
            'ids' : nodeIDList,
            'coords': xyList,
            'links' : source_target,
            'amount': number_node,
            'name_ID': nameIDList
            }

    return value

def get_id_name(tupleList,name):

    for tuple in tupleList:
        if tuple[0] == name:
            return tuple[1]


def main():
    root = tk.Tk()
    #pp = pprint.PrettyPrinter(indent=4)

    root.withdraw()

    xmlObject = filedialog.askopenfilename()
    xmlData = ET.parse(xmlObject).getroot()

    dictValue = get_links_with_coord(xmlData)
    source_target = dictValue['links']
    xyList = dictValue['coords']
    nodeIDList = dictValue['ids']
    nameIDList = dictValue['name_ID']
    n = int(dictValue['amount'])

    G = nt.Graph()

    # add the nodes based on the number of n
    G.addNodesFrom([nt.Node(f'{nodeIDList[index]}',f'{index}',xyList[index]) for index in range(n)])

    #add edges
    links = []

    for tuple in source_target:

        source = tuple[0]
        target = tuple[1]

        #using get_id_name to cconvert node name to id
        edge = nt.Edge(G.getNodes()[get_id_name(nameIDList,source)], G.getNodes()[get_id_name(nameIDList,target)])

        links.append(edge)


    G.addEdgesFrom(links)

    #########################################
    #plot the map in .gexf file, open using gephi network tool
    NodesList = G.getDict()['Nodes']
    EdgesList = G.getDict()['Edges']

    ######## gefx file
    g = nx.MultiGraph()

    for x in range(len(NodesList)):

        g.add_node(NodesList[x])

    for x in range(len(EdgesList)):

        tuple = EdgesList[x].get_tuple()
        g.add_edge(tuple[0],tuple[1])

    print(g.nodes())
    nx.write_gexf(g, "test.gexf", version="1.2draft")



main()
