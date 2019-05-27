#include "MySimulation.h"

Parser mySimulation :: createParser()
{
    Parser Germany50Parser;
    Germany50Parser.read_file();
    return Germany50Parser;
}

std::map<std::string, std::tuple<std::string, std::string>>
mySimulation :: getNodes(Parser g50)
{
    return g50.get_nodes();
}

std::map<std::string, std::tuple<std::string, std::string, std::string, std::string>>
mySimulation :: getLinks(Parser g50)
{
    return g50.get_links();
}


CommandLine mySimulation :: setup(int verbose, CommandLine commandLine)
{
    if (verbose == 1 || verbose == 2) {
        //LogComponentEnable("Germany50 Simulation", LOG_LEVEL_INFO);
    } else if (verbose == 2) {
        //LogComponentEnable("Germany50 Simulation", LOG_ALL);
    }

    Config::SetDefault("ns3::Ipv4GlobalRouting::RespondToInterfaceEvents", BooleanValue(true));
    Config::SetDefault("ns3::OnOffApplication::PacketSize", UintegerValue(210));
    Config::SetDefault("ns3::OnOffApplication::DataRate", StringValue("300b/s"));

    //commandLine.Parse(0,[]);
    return commandLine;
}

NodeContainer 
mySimulation :: createNodes(std::map<std::string, std::tuple<std::string, std::string>> simulationNodes)
{
    NodeContainer nestContainer;
    int numNodes = simulationNodes.size();
    nestContainer.Create(numNodes);
    return nestContainer;
}

NodeContainer 
mySimulation :: setNodes(NodeContainer container, std::map<std::string, std::tuple<std::string, std::string>> nodes)
{
    int i = 0;
    for (auto nodeIt = nodes.begin(); nodeIt != nodes.end(); ++nodeIt) {
        Ptr<Node> node = container.Get(i);

	    double latitude = std::stod(std::get<0>(nodeIt->second));
        double longitude = std::stod(std::get<1>(nodeIt->second));

        Names::Add(nodeIt->first, node);
        Vector coordinates = GeographicPositions::GeographicToCartesianCoordinates(
                latitude, longitude, 0, GeographicPositions::SPHERE);

        float x = (coordinates.x); 
        float y = (coordinates.y);
        
        AnimationInterface::SetConstantPosition(node, x, y);        
        ++i;
    }

    return container;
}

std::vector<NodeContainer> 
mySimulation :: connectNodes(std::map<std::string, std::tuple<std::string, std::string, std::string, std::string>> Links)
{
    std::vector<NodeContainer> subContainer(Links.size());

    int i = 0;
    for (auto linkIt = Links.begin(); linkIt != Links.end(); ++linkIt) {
        std::string source = std::get<0>(linkIt->second);
        std::string target = std::get<1>(linkIt->second);

        Ptr<Node> sourceNode (Names::Find<Node>(source));
        Ptr<Node> targetNode (Names::Find<Node>(target));
        
        subContainer[i] = NodeContainer(sourceNode, targetNode);
        
        ++i;
    }

    return subContainer;
}

std::tuple<std::vector<NodeContainer>,std::vector<NetDeviceContainer>>
mySimulation :: installP2P(std::vector<NodeContainer> SubContainers)
{
    PointToPointHelper p2p;

    p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    
    std::vector<NetDeviceContainer> P2PLinks (SubContainers.size());
    for(unsigned int i=0; i<SubContainers.size(); ++i) {
        P2PLinks[i] = p2p.Install(SubContainers[i]);
    } 
    std::tuple<std::vector<NodeContainer>,std::vector<NetDeviceContainer>> returnValues;
    std::get<0>(returnValues) = SubContainers;
    std::get<1>(returnValues) = P2PLinks;
    return returnValues;

}

std::vector<Ipv4InterfaceContainer>
mySimulation :: assignIP(NodeContainer cont, std::vector<NetDeviceContainer> p2pLinks)
{
    InternetStackHelper internet;
    internet.Install(cont);
    
    Ipv4AddressHelper ipv4;
    
    std::vector<Ipv4InterfaceContainer> ipv4List(p2pLinks.size());
    for(unsigned int i=0; i<p2pLinks.size(); ++i){
        std::ostringstream subnet;
        subnet<<"10.1."<<i+1<<".0";
        ipv4.SetBase(subnet.str().c_str(), "255.255.255.0");
        ipv4List[i] = ipv4.Assign(p2pLinks[i]);
    }

    return ipv4List;
}

void mySimulation :: populateRoutingTables()
{
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

std::vector<ns3::GlobalRoutingLSA>
mySimulation :: readLSA(std::vector<Ipv4InterfaceContainer> ipv4List)
{
    std::vector<ns3::GlobalRoutingLSA> lsaList(ipv4List.size());
//iterate thru the nodes
    int value = 0;
    
    NodeList::Iterator listEnd = NodeList::End ();
    
    for (NodeList::Iterator i = NodeList::Begin (); i != listEnd; i++)
    {
       //create a pointer for the current node
       Ptr<Node> node = *i;
       //check for GlobalRouter in the node
       Ptr<GlobalRouter> rtr = node->GetObject<GlobalRouter> ();
       //continue if theres no GlobalRouter
       if (!rtr)
         {
           std::cout << "Cant-be-Found ";      
           continue;
         }

       Ptr<Ipv4GlobalRouting> grouting = rtr->GetRoutingProtocol ();
       uint32_t numLSAs = rtr->DiscoverLSAs ();
       //std::cout << "Found " << numLSAs << " LSAs ";
       
       for (uint32_t j = 0; j < numLSAs; ++j)
        {
           ns3::GlobalRoutingLSA* lsa = new GlobalRoutingLSA ();
           //fetch a Link State Advertisement from the router
           rtr->GetLSA (j, *lsa);
           //std::cout << "Found:  " << *lsa ;
           //store in list
           lsaList[value] = *lsa;  
        }
        ++value; 
     }
    return lsaList;
}

void mySimulation :: execute()
{
    parser = createParser();
    nodes = getNodes(parser);
    links = getLinks(parser);
    cmd = setup(2,cmd);
    container = createNodes(nodes);
    container = setNodes(container,nodes);
    subcontainers = connectNodes(links);
    std::tuple<std::vector<NodeContainer>,std::vector<NetDeviceContainer>> output;
    output = installP2P(subcontainers);
    subcontainers = std::get<0>(output);
    p2pInstallations = std::get<1>(output);
    ip = assignIP(container,p2pInstallations);
    populateRoutingTables();
    grlsaList = readLSA(ip);
}




