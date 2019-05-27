import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog
from collections import namedtuple
import pprint

def get_nodes(xmlData):

    coordTupleList = []

    for node in xmlData[0][0]:
        nodeName = node.attrib.get('id')
        coordObj = node.find('./coordinates')
        xCoord = coordObj.find('x').text
        yCoord = coordObj.find('y').text
        coordTuple = {nodeName: (xCoord, yCoord)}
        coordTupleList.append(coordTuple)

    return coordTupleList

def get_links(xmlData):

    linkDataList = []

    for link in xmlData[0][1]:
        moduleDictList = []

        linkId = link.attrib.get('id')
        linkSource = link.find('source').text
        linkTarget = link.find('target').text
        linkModuleObjects = link.findall('additionalModules/addModule')
        for module in linkModuleObjects:
            capacity = module.find('capacity').text
            cost = module.find('cost').text
            moduleDict = {'capacity': capacity, 'cost': cost}
            moduleDictList.append(moduleDict)

        #linkCapacity = linkModuleList
        linkTuple = {linkId: (linkSource, linkTarget)}
        linkModulesTuple = (linkTuple, moduleDictList)
        linkDataList.append(linkModulesTuple)

    return linkDataList

def main():
    root = tk.Tk()
    pp = pprint.PrettyPrinter(indent=4)
    
    root.withdraw()

    xmlObject = filedialog.askopenfilename()
    xmlData = ET.parse(xmlObject).getroot()

    coordTupleList = get_nodes(xmlData)
    linkDataList = get_links(xmlData)

    pp.pprint(coordTupleList)
    pp.pprint(linkDataList)

main()