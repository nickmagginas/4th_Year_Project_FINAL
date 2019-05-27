#include <map>
#include <vector>
#include "parser.h"
#include "ns3/vector.h"
#include "ns3/ptr.h"
#include "ns3/core-module.h"
#include "ns3/global-route-manager.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/bridge-module.h"
#include "ns3/csma-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/point-to-point-grid.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/flow-monitor.h"
#include "ns3/log.h"
#include "ns3/geographic-positions.h"
#include "ns3/position-allocator.h"
#include "ns3/names.h"

#include "ns3/assert.h"
#include "ns3/simulation-singleton.h"
#include "ns3/global-route-manager-impl.h"

using namespace ns3;

class mySimulation
{
    public:
    std::map<std::string, std::tuple<std::string, std::string>> nodes;
    std::map<std::string, std::tuple<std::string, std::string, std::string, std::string>> links;
    NodeContainer container;
    std::vector<NodeContainer> subcontainers;
    std::vector<NetDeviceContainer> p2pInstallations;
    std::vector<Ipv4InterfaceContainer> ip;
    Parser parser;
    CommandLine cmd;
    std::vector<ns3::GlobalRoutingLSA> grlsaList;

    Parser 
    createParser();
    std::map<std::string, std::tuple<std::string, std::string>> 
    getNodes(Parser parser);  
    std::map<std::string, std::tuple<std::string, std::string, std::string, std::string>> 
    getLinks(Parser parser);
    CommandLine 
    setup(int verbose, CommandLine commandLine);
    NodeContainer 
    createNodes(std::map<std::string, std::tuple<std::string, std::string>> nodes);
    NodeContainer 
    setNodes(NodeContainer container,std::map<std::string, std::tuple<std::string, std::string>> nodes);
    std::vector<NodeContainer> 
    connectNodes(std::map<std::string, std::tuple<std::string, std::string, std::string, std::string>> Links);
    std::tuple<std::vector<NodeContainer>,std::vector<NetDeviceContainer>> 
    installP2P( std::vector<NodeContainer> SubContainers);
    std::vector<Ipv4InterfaceContainer> 
    assignIP(NodeContainer cont, std::vector<NetDeviceContainer> p2pLinks);
    void 
    populateRoutingTables();
    std::vector<ns3::GlobalRoutingLSA> 
    readLSA(std::vector<Ipv4InterfaceContainer> ipv4List);
    void 
    execute();

};