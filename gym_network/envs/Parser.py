import xml.etree.ElementTree as ET


class Parser:
    def __init__(self, filename):
        # xml files now stored in 'network_xmls' directory:
        filepath = "./network_xmls/" + filename
        self.root = ET.parse(filepath).getroot()
        self.sections = [child for child in self.root]
        self.nodes, self.links = self._getData()

    def _getData(self):
        [nodes, links] = [child for child in self.sections[0]]
        # keys contains the 'id's of the Germany50 nodes (e.g. 'Berlin')
        keys = []
        # iterate through all the nodes listed in xml file and append their id's to keys list
        for child in nodes:
            keys = keys + [child.attrib["id"]]
        node_data = {key: None for key in keys}
        # get coordinates for each node:
        attributes = [
            [axis.text for axis in coordinate] for node in nodes for coordinate in node
        ]
        for index, key in enumerate(node_data.keys()):
            node_data[key] = attributes[index]
        link_data = {
            key: None for key in [attributes.attrib["id"] for attributes in links]
        }

        attributes = [
            [
                axis.text
                for axis in link
                if any(axis.tag == tag for tag in ["source", "target"])
            ]
            for link in links
        ]
        # iterate through each link and get the capacity under "additionalModules/addModule/capacity", and get the text from the capacity object
        # appends each capacity to capacity list which is returned seperately
        for idx, link in enumerate(links):
            #note the find method only finds the FIRST matching path. Some networks have multiple 'addModule'
            #but only the first one is obtained.
            attributes[idx].extend(
                [
                    idx, #index of link
                    link.find("additionalModules/addModule/capacity").text, #capacity
                    link.find("additionalModules/addModule/cost").text, #cost
                ]
            )
        for index, key in enumerate(link_data.keys()):
            link_data[key] = attributes[index]
        return node_data, link_data
